#!/usr/bin/env python3
"""mtrf_singlelag.cpp の出力(単一ラグごとの予測相関r)を、全被験者・全単語で平均して1枚の図にする。

入力 (C++側の出力):
  <data_dir>/<tag>/<subj>/mtrf_rlag_<word>.npy ... shape (layer, channel, lag) のtest r
  (ラグ軸: lag_ms = TMIN_MS + LAG_STEP_MS * index)

出力:
  <out_dir>/all_average_singlelag.png ... 6パネルの図 (1枚)
  <out_dir>/summary_singlelag.csv     ... r最大ラグ (ALL/被験者別 × layer平均/各layer)

「ピーク」の定義 (先行研究と同じ):
  単一ラグモデルの予測相関rが最大になったラグ。TRFの重みのピークではない。

平均の取り方: 1ユニット = 被験者 × 単語条件(例: kk/ame_d)
  - まずユニットごとにchannel平均 → ユニット間で平均 (帯は SEM)
  - ピークは「ユニット平均した曲線」のargmaxと、「ユニットごとのargmax」の分布の両方を見る

使い方:
  python plot_mtrf_singlelag.py --data-dir ../data --tag 1009_singlelag --out figures
  (--smooth-ms 20 で曲線を移動平均してからピークを取る。既定は平滑化なし)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# C++側の設定と合わせること (tmin=0, lag_step=1)
TMIN_MS = 0
LAG_STEP_MS = 1


def load_units(data_dir: Path, tag: str):
    units = []
    for p in sorted((data_dir / tag).glob("*/mtrf_rlag_*.npy")):
        units.append({
            "subject": p.parent.name,
            "unit": f"{p.parent.name}/{p.stem[len('mtrf_rlag_'):]}",
            "rlag": np.load(p),  # (layer, channel, lag)
        })
    if not units:
        raise SystemExit(f"{data_dir}/{tag}/*/mtrf_rlag_*.npy が見つかりません")
    return units


def detect_layers(rlag: np.ndarray) -> list:
    """実際に計算されたlayer (全部0/NaNでないもの)"""
    s = np.abs(np.nan_to_num(rlag)).sum(axis=(1, 2))
    return [int(i) for i in np.where(s > 0)[0]]


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """最終軸方向の移動平均 (端は端の値で延長)。w<=1なら何もしない"""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, w - 1 - pad)], mode="edge")
    k = np.ones(w) / w
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), -1, xp)
    return out


def sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    return x.std(axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(x.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--tag", default="1009_singlelag", help="C++側の today に対応")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--smooth-ms", type=int, default=1, help="ピーク検出・描画前の移動平均窓[ms] (1=なし)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = load_units(Path(args.data_dir), args.tag)
    layers = detect_layers(units[0]["rlag"])
    n_lags = units[0]["rlag"].shape[2]
    lags = TMIN_MS + LAG_STEP_MS * np.arange(n_lags)
    w = max(1, args.smooth_ms // LAG_STEP_MS)
    subjects = sorted({u["subject"] for u in units})
    print(f"units={len(units)}, layers={layers}, n_lags={n_lags} ({lags[0]}..{lags[-1]}ms), smooth={w}")

    # R: (unit, layer, channel, lag)
    R = np.stack([u["rlag"][layers] for u in units])
    R = smooth(R, w)
    Rc = np.nanmean(R, axis=2)           # (unit, layer, lag): channel平均
    Rl = np.nanmean(Rc, axis=1)          # (unit, lag): layer平均
    subj_of = np.array([u["subject"] for u in units])

    def peak_ms(curve):
        return float(lags[np.nanargmax(curve)])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Single-lag model: test r vs lag  (all subjects / words, n = {len(units)} units)", fontsize=14)

    # (0,0) layerごとの r vs lag (ユニット平均・channel平均)
    ax = axes[0, 0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for cl, li, layer in zip(cmap, range(len(layers)), layers):
        curve = Rc[:, li].mean(axis=0)
        ax.plot(lags, curve, color=cl, label=f"L{layer}")
        pk = np.nanargmax(curve)
        ax.plot(lags[pk], curve[pk], "o", color=cl)
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Test r (mean over channels & units)")
    ax.set_title("r vs lag per layer  (dot = peak)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # (0,1) layer平均の r vs lag (帯=SEM over units)
    ax = axes[0, 1]
    mean_curve = Rl.mean(axis=0)
    band = sem(Rl, axis=0)
    ax.plot(lags, mean_curve, color="tab:blue")
    ax.fill_between(lags, mean_curve - band, mean_curve + band, color="tab:blue", alpha=0.25)
    pk = peak_ms(mean_curve)
    ax.axvline(pk, color="tab:red", ls="--", label=f"peak = {pk:.0f} ms")
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Test r (mean over layers, channels)")
    ax.set_title("Overall r vs lag  (band = SEM over units)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,2) layerごとのピークラグ: ユニット平均曲線のargmax と ユニットごとのargmaxの分布
    ax = axes[0, 2]
    peaks_avg = [peak_ms(Rc[:, li].mean(axis=0)) for li in range(len(layers))]
    unit_peaks = np.array([[peak_ms(Rc[u, li]) for li in range(len(layers))] for u in range(len(units))])
    med = np.median(unit_peaks, axis=0)
    q1, q3 = np.percentile(unit_peaks, [25, 75], axis=0)
    xs = np.arange(len(layers))
    ax.fill_between(xs, q1, q3, color="gray", alpha=0.25, label="IQR over units")
    ax.plot(xs, med, "s-", color="gray", label="median of per-unit peaks")
    ax.plot(xs, peaks_avg, "o-", color="tab:red", label="peak of unit-averaged curve")
    ax.set_xticks(xs)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Peak lag [ms]")
    ax.set_title("Lag with max r, per layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) 被験者別の r vs lag (layer平均)
    ax = axes[1, 0]
    for s, c in zip(subjects, plt.cm.tab10(np.arange(len(subjects)))):
        curve = Rl[subj_of == s].mean(axis=0)
        pk = peak_ms(curve)
        ax.plot(lags, curve, color=c, label=f"{s} (peak {pk:.0f} ms)")
        ax.axvline(pk, color=c, ls=":", lw=0.8)
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Test r (mean over layers, channels)")
    ax.set_title("Per subject")
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,1) layer × lag のヒートマップ
    ax = axes[1, 1]
    M = Rc.mean(axis=0)  # (layer, lag)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                   extent=[lags[0], lags[-1], -0.5, len(layers) - 0.5])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Layer")
    ax.set_title("r (unit & channel averaged)")
    fig.colorbar(im, ax=ax, label="r")

    # (1,2) channelごと・ユニットごとのピークラグの分布 (layer平均曲線ベース)
    ax = axes[1, 2]
    Rch = np.nanmean(R, axis=1)  # (unit, channel, lag): layer平均
    ch_peaks = lags[np.nanargmax(np.nan_to_num(Rch, nan=-np.inf), axis=2)].ravel()
    bins = np.arange(lags[0], lags[-1] + 26, 25)
    ax.hist(ch_peaks, bins=bins, color="tab:gray")
    ax.axvline(np.median(ch_peaks), color="tab:red", ls="--", label=f"median = {np.median(ch_peaks):.0f} ms")
    ax.set_xlabel("Peak lag [ms]")
    ax.set_ylabel("Count (unit x channel)")
    ax.set_title("Per unit x channel peak (layer-averaged curve)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "all_average_singlelag.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    # summary: ALL / 被験者別 × layer平均 / 各layer
    rows = []
    for scope in ["ALL"] + subjects:
        mask = np.ones(len(units), bool) if scope == "ALL" else (subj_of == scope)
        rows.append({"scope": scope, "layer": "avg", "n_units": int(mask.sum()),
                     "peak_lag_ms": peak_ms(Rl[mask].mean(axis=0)),
                     "peak_r": float(np.nanmax(Rl[mask].mean(axis=0)))})
        for li, layer in enumerate(layers):
            curve = Rc[mask, li].mean(axis=0)
            rows.append({"scope": scope, "layer": layer, "n_units": int(mask.sum()),
                         "peak_lag_ms": peak_ms(curve), "peak_r": float(np.nanmax(curve))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_singlelag.csv", index=False)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nsaved {out_dir / 'summary_singlelag.csv'}")


if __name__ == "__main__":
    main()