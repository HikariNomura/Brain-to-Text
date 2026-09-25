#!/usr/bin/env python3
"""
Whisper特徴量に対するPCAの次元数(n_components)の妥当性を確認するスクリプト．

各層(layer)ごとに、全単語×全韻律条件(8条件)を1つの学習データとしてPCAをフルにかけ、
- 各主成分の寄与率(explained variance ratio)
- 累積寄与率
- n_components=10 での累積寄与率
- 累積寄与率が 90%/95%/99% に達するのに必要な次元数
を算出・可視化する．

使い方:
    python scripts/check_pca_dimensionality.py "C:/.../Whisper/Feature_extraction"
    python scripts/check_pca_dimensionality.py "..." --layers 0 10 20 31
    python scripts/check_pca_dimensionality.py "..." --n_check 5 10 20 50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


WORD_LIST = ["ame", "nizi", "momo", "hasi"]
OTO = ["_d", "_u"]
AUDIO_LENGTH_SEC = 3.0
WHISPER_SR = 50

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


def load_feature(whisper_root: Path, word: str, prosody: str, layer: int) -> np.ndarray:
    """(frames, 1280) の特徴量を返す"""
    path = whisper_root / f"{word}{prosody}" / f"layer_{layer}.npy"
    arr = np.load(path)
    arr = np.squeeze(arr)
    valid_frames = int(AUDIO_LENGTH_SEC * WHISPER_SR)
    if arr.shape[0] < valid_frames:
        raise ValueError(f"{path}: frames({arr.shape[0]}) < valid_frames({valid_frames})")
    return arr[:valid_frames]


def load_layer_data(whisper_root: Path, layer: int) -> np.ndarray:
    """指定レイヤーについて、全単語×全韻律条件のフレームを縦に結合して返す
    (n_samples, 1280)。n_samples = 8条件 × フレーム数"""
    feats = []
    for word in WORD_LIST:
        for prosody in OTO:
            feats.append(load_feature(whisper_root, word, prosody, layer))
    return np.vstack(feats)


def explained_variance_at_k(evr_cumsum: np.ndarray, k: int) -> float:
    """k次元(1-indexed)までの累積寄与率を返す"""
    idx = min(k, len(evr_cumsum)) - 1
    return float(evr_cumsum[idx])


def n_components_for_threshold(evr_cumsum: np.ndarray, threshold: float) -> int:
    """累積寄与率がthresholdに初めて達する次元数(1-indexed)を返す"""
    idx = np.searchsorted(evr_cumsum, threshold)
    if idx >= len(evr_cumsum):
        return len(evr_cumsum)
    return idx + 1


def main():
    parser = argparse.ArgumentParser(
        description="Whisper特徴量に対するPCA次元数(n_components)の妥当性を確認する"
    )
    parser.add_argument("whisper_root", type=str, help="Whisper特徴量のルートディレクトリ")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                         help="確認するレイヤー番号のリスト(省略時は0,4,8,...,31を等間隔で抽出)")
    parser.add_argument("--n_all_layers", type=int, default=32, help="全レイヤー数")
    parser.add_argument("--current_n", type=int, default=10,
                         help="現在使っている次元数(比較の基準)")
    parser.add_argument("--n_check", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20, 30, 50],
                         help="累積寄与率を確認する次元数の候補リスト")
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    whisper_root = Path(args.whisper_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.layers is None:
        layers_to_check = list(range(0, args.n_all_layers, 4))  # 0,4,8,...,28程度を等間隔で
        if (args.n_all_layers - 1) not in layers_to_check:
            layers_to_check.append(args.n_all_layers - 1)
    else:
        layers_to_check = args.layers

    print(f"確認するレイヤー: {layers_to_check}")
    print(f"現在の次元数: {args.current_n}\n")

    all_evr = {}          # layer -> explained_variance_ratio_ (全次元分)
    records = []          # 表形式のサマリ用

    max_components = None

    for layer in layers_to_check:
        X = load_layer_data(whisper_root, layer)  # (n_samples, 1280)
        n_components_full = min(X.shape[0], X.shape[1])  # PCAは min(n_samples, n_features) が上限
        if max_components is None:
            max_components = n_components_full

        pca = PCA(n_components=n_components_full).fit(X)
        evr = pca.explained_variance_ratio_
        evr_cumsum = np.cumsum(evr)
        all_evr[layer] = evr_cumsum

        current_cum = explained_variance_at_k(evr_cumsum, args.current_n)
        n90 = n_components_for_threshold(evr_cumsum, 0.90)
        n95 = n_components_for_threshold(evr_cumsum, 0.95)
        n99 = n_components_for_threshold(evr_cumsum, 0.99)

        print(f"Layer {layer:2d}: n_components={args.current_n} → 累積寄与率 {current_cum*100:.2f}%  "
              f"| 90%到達: {n90}次元, 95%到達: {n95}次元, 99%到達: {n99}次元")

        record = {"layer": layer, f"cum_var_at_{args.current_n}": current_cum,
                   "n_for_90pct": n90, "n_for_95pct": n95, "n_for_99pct": n99}
        for k in args.n_check:
            record[f"cum_var_at_{k}"] = explained_variance_at_k(evr_cumsum, k)
        records.append(record)

    df = pd.DataFrame(records)
    csv_path = outdir / "pca_dimensionality_check.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # =====================================================
    # 図1: 累積寄与率カーブ(レイヤーごとに重ねる) + 現在のn_componentsに縦線
    # =====================================================
    fig, ax = plt.subplots(figsize=(11, 6))

    show_dims = min(60, max_components)  # 見やすさのため、最初の60次元程度まで表示
    for layer in layers_to_check:
        evr_cumsum = all_evr[layer]
        ax.plot(np.arange(1, show_dims + 1), evr_cumsum[:show_dims] * 100,
                marker="o", markersize=3, linewidth=1.5, label=f"Layer {layer}")

    ax.axvline(args.current_n, color="red", linestyle="--", linewidth=2,
               label=f"Current n_components = {args.current_n}")
    ax.axhline(90, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, ncol=2)

    fig.tight_layout()
    fig_path = outdir / "pca_cumulative_variance.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # =====================================================
    # 図2: n_componentsの候補ごとに、累積寄与率がどう変わるか(棒グラフ、レイヤー平均)
    # =====================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    mean_cum_by_k = []
    for k in args.n_check:
        vals = [explained_variance_at_k(all_evr[layer], k) for layer in layers_to_check]
        mean_cum_by_k.append(np.mean(vals) * 100)

    bars = ax2.bar([str(k) for k in args.n_check], mean_cum_by_k, color="tab:blue", alpha=0.8)
    for bar, k in zip(bars, args.n_check):
        if k == args.current_n:
            bar.set_color("tab:red")

    ax2.set_xlabel("Number of components (n_components)")
    ax2.set_ylabel("Mean cumulative explained variance (%)\n(averaged across checked layers)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(90, color="gray", linestyle=":", linewidth=1)
    ax2.axhline(95, color="gray", linestyle=":", linewidth=1)

    fig2.tight_layout()
    fig2_path = outdir / "pca_ncomponents_comparison.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # =====================================================
    # サマリ表示
    # =====================================================
    print("\n=== サマリ(全確認レイヤーの平均) ===")
    for k in args.n_check:
        vals = [explained_variance_at_k(all_evr[layer], k) for layer in layers_to_check]
        marker = " <- 現在の設定" if k == args.current_n else ""
        print(f"n_components={k:3d}: 平均累積寄与率 = {np.mean(vals)*100:.2f}%  "
              f"(範囲: {np.min(vals)*100:.2f}%〜{np.max(vals)*100:.2f}%){marker}")


if __name__ == "__main__":
    main()