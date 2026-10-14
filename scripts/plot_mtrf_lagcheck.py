#!/usr/bin/env python3
"""mtrf_lagcheck.cpp の出力を、全被験者・全単語(_d/_u込み)で平均して1枚の図にする。

入力 (C++側の出力):
  <data_dir>/<tag>_tmax<ms>/<subj>/mtrf_agg_<word>.csv        ... layer×channelごとの r / peak_lag_ms
  <data_dir>/<tag>_tmax<ms>/<subj>/mtrf_lagprofile_<word>.npy ... (layer, channel, lag) の重みの大きさ

出力:
  <out_dir>/all_average.png ... 6パネルの図 (1枚)
  <out_dir>/summary.csv     ... tmaxごとの平均r・ピークラグ中央値・端張り付き率 (ALL と被験者別)

平均の取り方:
  「1ユニット = 被験者 × 単語条件(例: kk/ame_d)」とし、
  - r         : ユニット内でchannel平均 → ユニット間で 平均 ± SEM
  - ピークラグ : 外れ値に強いよう中央値を使用 (ユニット内channel中央値 → ユニット間の中央値・四分位)
  - プロファイル: ユニットごとにchannel平均して最大値で正規化 → ユニット間で平均

使い方:
  python plot_mtrf_lagcheck.py --data-dir ../data --tag 1009_lagcheck --out figures
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# C++側の設定と合わせること (tmin=0, lag_step=10)
TMIN_MS = 0
LAG_STEP_MS = 10

# ラグプロファイルを描く tmax [ms]
TOP_PROFILE_TMAXS = [150, 175]      # 上段の中央・右
PROFILE_TMAXS = [200, 500, 750]     # 下段
ALIGN_ALL_PROFILES = False          # True: 上段2枚+下段3枚の計5枚でy軸の範囲を揃える
                                    # False: 下段3枚どうし・上段2枚どうしで揃える


def find_runs(data_dir: Path, tag: str) -> dict:
    """{tmax_ms: ディレクトリ} を tmax の昇順で返す"""
    runs = {}
    for d in data_dir.glob(f"{tag}_tmax*"):
        m = re.fullmatch(rf"{re.escape(tag)}_tmax(\d+)", d.name)
        if m and d.is_dir():
            runs[int(m.group(1))] = d
    return dict(sorted(runs.items()))


def load_units(runs: dict):
    """全tmaxのファイルが揃っているユニットだけ読み込む。
    戻り値: (tmaxごとの結合DataFrame {tm: df(unit,subject列付き)}, プロファイル {tm: {unit: array}})"""
    first = next(iter(runs))
    frames = {tm: [] for tm in runs}
    profs = {tm: {} for tm in runs}
    n_units = 0
    for agg_path in sorted(runs[first].glob("*/mtrf_agg_*.csv")):
        subj = agg_path.parent.name
        word = agg_path.stem[len("mtrf_agg_"):]
        unit = f"{subj}/{word}"
        paths = {tm: d / subj / f"mtrf_agg_{word}.csv" for tm, d in runs.items()}
        if not all(p.exists() for p in paths.values()):
            print(f"  [skip] {unit}: tmaxの一部が欠けています")
            continue
        for tm, p in paths.items():
            df = pd.read_csv(p)
            df["unit"], df["subject"] = unit, subj
            frames[tm].append(df)
            pp = runs[tm] / subj / f"mtrf_lagprofile_{word}.npy"
            if pp.exists():
                profs[tm][unit] = np.load(pp)
        n_units += 1
    if n_units == 0:
        raise SystemExit("読み込めるユニットがありません")
    print(f"units: {n_units}")
    return {tm: pd.concat(fs, ignore_index=True) for tm, fs in frames.items()}, profs, n_units


def lag_axis(n_lags: int, tmax_ms: int) -> np.ndarray:
    ax = TMIN_MS + np.arange(n_lags) * LAG_STEP_MS
    if ax[-1] != tmax_ms:
        print(f"[warn] lag axis end ({ax[-1]}ms) != tmax ({tmax_ms}ms): TMIN_MS/LAG_STEP_MS を確認")
    return ax


def sem(x):
    x = np.asarray(x, dtype=float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def plot_lag_profile(ax, profs_tm, layers, tm):
    """ユニットごとにchannel平均→最大値で正規化→ユニット間平均、をlayerごとの線で描く"""
    if not profs_tm:
        ax.text(0.5, 0.5, "lagprofile.npy not found", ha="center")
        ax.grid(alpha=0.3)
        return
    n_lags = next(iter(profs_tm.values())).shape[2]
    lags = lag_axis(n_lags, tm)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for cl, layer in zip(cmap, layers):
        ps = []
        for prof in profs_tm.values():
            p = prof[layer].mean(axis=0)
            if p.max() > 0:
                ps.append(p / p.max())
        if ps:
            ax.plot(lags, np.mean(ps, axis=0), color=cl, label=f"L{layer}")
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Weight norm (normalized per unit & layer, mean over units)")
    ax.set_title(f"TRF lag profile (0-{tm}ms)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--tag", default="1009_lagcheck", help="C++側の today に対応")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs(data_dir, args.tag)
    if not runs:
        raise SystemExit(f"{data_dir}/{args.tag}_tmax* が見つかりません")
    print("tmax runs:", {k: str(v) for k, v in runs.items()})

    aggs, profs, n_units = load_units(runs)
    tmaxs = list(runs.keys())
    first = tmaxs[0]
    color_of = {tm: c for tm, c in zip(tmaxs, plt.cm.tab10(np.arange(len(tmaxs))))}
    layers = sorted(aggs[first]["layer"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"All subjects / words averaged  (n = {n_units} subject x word units)", fontsize=14)

    # (上・左) layerごとの相関r: ユニット内channel平均 → ユニット間 平均±SEM
    ax = axes[0, 0]
    for tm in tmaxs:
        u = aggs[tm].groupby(["unit", "layer"])["mean_r"].mean().unstack("layer")  # unit x layer
        ax.errorbar(u.columns, u.mean(axis=0).values, yerr=[sem(u[l]) for l in u.columns],
                    marker="o", capsize=3, color=color_of[tm], label=f"0-{tm}ms")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation r (channel mean; bar = SEM over units)")
    ax.set_title("Test correlation vs layer")
    ax.legend()
    ax.grid(alpha=0.3)

    # ラグプロファイル: 上段(中央・右)と下段(3枚)
    top_axes = [axes[0, 1], axes[0, 2]]
    bottom_axes = list(axes[1])
    placements = list(zip(top_axes, TOP_PROFILE_TMAXS)) + list(zip(bottom_axes, PROFILE_TMAXS))
    for ax, tm in placements:
        if tm in profs and profs[tm]:
            plot_lag_profile(ax, profs[tm], layers, tm)
        else:
            ax.text(0.5, 0.5, f"tmax={tm}ms の結果がありません\n(C++側の tmax_list に追加して再実行)",
                    ha="center", va="center")
            ax.set_title(f"TRF lag profile (0-{tm}ms)")
            ax.grid(alpha=0.3)

    # y軸の範囲を揃える (データのある枠だけで最小〜最大を取る)
    def align_y(pairs):
        data_axes = [ax for ax, tm in pairs if tm in profs and profs[tm]]
        if data_axes:
            ymin = min(ax.get_ylim()[0] for ax in data_axes)
            ymax = max(ax.get_ylim()[1] for ax in data_axes)
            for ax in data_axes:
                ax.set_ylim(ymin, ymax)

    if ALIGN_ALL_PROFILES:
        align_y(placements)
    else:
        align_y(list(zip(top_axes, TOP_PROFILE_TMAXS)))
        align_y(list(zip(bottom_axes, PROFILE_TMAXS)))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "all_average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # summary.csv: ALL と被験者別
    rows = []
    for tm in tmaxs:
        a = aggs[tm]
        groups = [("ALL", a)] + [(s, g) for s, g in a.groupby("subject")]
        for name, g in groups:
            rows.append({
                "subject": name, "tmax_ms": tm, "n_units": g["unit"].nunique(),
                "mean_r": g["mean_r"].mean(),
                "median_peak_lag_ms": g["peak_lag_ms"].median(),
                "edge_fraction": float(np.mean(g["peak_lag_ms"].values >= tm)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nsaved {out_dir / 'summary.csv'}")
    print("edge_fraction が大きい(例: >0.3)tmaxは、ピークが探索範囲の端に張り付いている可能性があります。")


if __name__ == "__main__":
    main()