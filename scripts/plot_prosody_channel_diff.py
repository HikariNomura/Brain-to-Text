#!/usr/bin/env python3
"""
韻律条件間（_d vs _u）のスカログラム差分を、チャンネル配置図で可視化するスクリプト．

main.cpp が出力する mtrf_pred_<word>.npy / mtrf_test_<word>.npy
（形状: [32層, 21ch, 1000samples]）を読み込み，
全被験者・全単語で平均した後，_d と _u のスカログラム（CWT）の正規化パワー差分の RMS を
チャンネル配置図（5行×6列）として可視化する．

レイヤーは以下の3パターンで平均する:
    - 全レイヤー平均 (0-31)
    - 浅い層平均 (0-3)
    - 深い層平均 (20-31)

予測脳波（pred）と実測脳波（test）それぞれについて、3パネル(全平均/浅い層/深い層)の図を出力する．

使い方:
    python scripts/plot_prosody_channel_diff.py
    python scripts/plot_prosody_channel_diff.py --data-root ../data/0913
    python scripts/plot_prosody_channel_diff.py --today 0913 --output-root ../result
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import morlet2
from scipy.signal import fftconvolve
from pathlib import Path
import argparse


# ---- 設定（main.cpp の定数と対応） ----
SUBJECTS   = ["kk", "tr", "ks"]
WORD_LIST  = ["ame", "nizi", "momo", "hasi"]
N_LAYERS   = 32   # layer 0..31
N_CHANNELS = 21
EEG_SR     = 1000  # Hz
TODAY      = "0913"

SHALLOW_LAYERS = range(0, 4)    # 0-3
DEEP_LAYERS = range(20, 32)     # 20-31

# プロット設定: (ファイル接頭辞, 凡例ラベル)
SOURCES = [
    ("mtrf_pred", "predicted"),
    ("mtrf_test", "raw EEG"),
]

# チャンネル配置 (5行 x 6列)。空欄は None。
GRID_LAYOUT = [
    [7,    8,  None, 17, None, None],
    [6,    9,    15, 18, None, None],
    [5,   10,    14, 19,    1,   16],
    [4,   11,    13, 20,    2,    3],
    [None, 12, None, 21, None, None],
]

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


def _cwt(data, wavelet, widths, w):
    """scipy.signal.cwt の代替実装"""
    output = np.empty((len(widths),) + data.shape, dtype=np.complex128)
    for ind, width in enumerate(widths):
        N = int(np.min([10 * width, len(data)]))
        wavelet_data = np.conj(wavelet(N, width, w=w)[::-1])
        output[ind] = fftconvolve(data, wavelet_data, mode="same")
    return output


def compute_scalogram(signal, sr=1000, freqs=None):
    """CWT（連続ウェーブレット変換）によるスカログラムを計算する"""
    if freqs is None:
        freqs = np.linspace(1, 40, 80)

    w = 6.0
    widths = w * sr / (2 * np.pi * freqs)
    cwtm = _cwt(signal, morlet2, widths, w=w)
    power = np.abs(cwtm) ** 2

    mean_power = power.mean(axis=1, keepdims=True)
    mean_power[mean_power == 0] = 1.0
    power_norm = power / mean_power

    return power_norm


def compute_diff_rms_per_channel(data_root, prefix, freqs):
    """
    指定した接頭辞（mtrf_pred / mtrf_test）について，
    全被験者・全単語平均の「チャンネル×レイヤー」ごとスカログラム差分RMSを計算する．

    Returns
    -------
    diff_rms : (N_LAYERS, N_CHANNELS) array もしくは None（有効データなしの場合）
    n_valid : 有効な (被験者, 単語) ペア数
    """
    diff_rms = np.zeros((N_LAYERS, N_CHANNELS))
    n_valid = 0

    for subj in SUBJECTS:
        for word in WORD_LIST:
            fname_d = f"{prefix}_{word}_d.npy"
            fname_u = f"{prefix}_{word}_u.npy"

            path_d = data_root / subj / fname_d
            path_u = data_root / subj / fname_u

            if not path_d.exists() or not path_u.exists():
                print(f"  [SKIP] ({prefix}) {subj}/{word}: ファイルが見つかりません")
                continue

            data_d = np.load(str(path_d))  # (32, 21, 1000)
            data_u = np.load(str(path_u))  # (32, 21, 1000)
            print(f"  [OK]   ({prefix}) {subj}/{word}")

            for layer in range(N_LAYERS):
                for ch in range(N_CHANNELS):
                    sig_d = data_d[layer, ch, :]
                    sig_u = data_u[layer, ch, :]

                    scalo_d = compute_scalogram(sig_d, sr=EEG_SR, freqs=freqs)
                    scalo_u = compute_scalogram(sig_u, sr=EEG_SR, freqs=freqs)

                    diff = scalo_d - scalo_u
                    rms = np.sqrt(np.mean(diff ** 2))
                    diff_rms[layer, ch] += rms

            n_valid += 1

    if n_valid == 0:
        return None, 0

    diff_rms /= n_valid
    return diff_rms, n_valid


def plot_channel_layout(diff_rms, label, output_path):
    """diff_rms (N_LAYERS, N_CHANNELS) から、全平均/浅い層/深い層の3パネルを描画する"""
    n_rows = len(GRID_LAYOUT)
    n_cols = len(GRID_LAYOUT[0])

    all_mean = diff_rms.mean(axis=0)                      # (N_CHANNELS,)
    shallow_mean = diff_rms[list(SHALLOW_LAYERS)].mean(axis=0)
    deep_mean = diff_rms[list(DEEP_LAYERS)].mean(axis=0)

    panels = [
        ("All layers (0-31)", all_mean),
        ("Shallow layers (0-3)", shallow_mean),
        ("Deep layers (20-31)", deep_mean),
    ]

    vmin = min(p[1].min() for p in panels)
    vmax = max(p[1].max() for p in panels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im = None
    for ax, (title, ch_values) in zip(axes, panels):
        grid_values = np.full((n_rows, n_cols), np.nan)
        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None:
                    # チャンネル番号(1始まり)をindex(0始まり)に変換
                    grid_values[i, j] = ch_values[ch - 1]

        masked = np.ma.masked_invalid(grid_values)
        im = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax)

        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None:
                    val = grid_values[i, j]
                    ax.text(j, i, f"{ch}\n{val:.3f}", ha="center", va="center",
                            fontsize=8, color="white")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.7,
                 label="Scalogram Difference RMS\n(Normalized Power, H-L vs L-H)")
    fig.suptitle(f"Prosody Contrast (H-L vs L-H) by Channel: {label}", fontsize=16)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"画像を保存しました: {output_path.resolve()}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="韻律条件間（_d vs _u）のスカログラム差分RMSを，チャンネル配置図で可視化")
    parser.add_argument("--data-root", type=str, default=None,
                        help="データディレクトリのルート（デフォルト: ./data/<TODAY>）")
    parser.add_argument("--today", type=str, default=TODAY,
                        help="日付ラベル（デフォルト: 0913）")
    parser.add_argument("--output-root", type=str, default="./result",
                        help="出力先ルートディレクトリ（デフォルト: ./result）")
    args = parser.parse_args()

    if args.data_root is None:
        data_root = Path("./data") / args.today
    else:
        data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"エラー: データディレクトリが見つかりません: {data_root}")
        return

    print(f"データルート: {data_root.resolve()}")

    freqs = np.linspace(1, 40, 80)

    output_dir = Path(args.output_root) / args.today
    output_dir.mkdir(parents=True, exist_ok=True)

    for prefix, label in SOURCES:
        print(f"\n--- {label} ({prefix}) を計算中 ---")
        diff_rms, n_valid = compute_diff_rms_per_channel(data_root, prefix, freqs)
        if diff_rms is None:
            print(f"警告: {prefix} の有効なデータが見つかりませんでした．スキップします．")
            continue
        print(f"有効な (被験者, 単語) ペア数: {n_valid}")

        output_path = output_dir / f"prosody_channel_diff_{prefix}.png"
        plot_channel_layout(diff_rms, label, output_path)


if __name__ == "__main__":
    main()