#!/usr/bin/env python3
"""
韻律条件間（_d vs _u）のスカログラム差分を可視化するスクリプト．

main.cpp が出力する mtrf_pred_<word>.npy / mtrf_test_<word>.npy
（形状: [32層, 21ch, 1000samples]）を読み込み，
全被験者・全単語で平均した後，_d と _u のスカログラム（CWT）の正規化パワー差分の RMS を
横軸＝Whisperレイヤー，縦軸＝差分RMS としてプロットする．

予測脳波（pred）と実測脳波（test）の結果を同一グラフ上に重ねて表示し，
result/<today>/ 以下に自動保存する．

使い方:
    python scripts/plot_prosody_scalogram_diff.py
    python scripts/plot_prosody_scalogram_diff.py --data-root ../data/0913
    python scripts/plot_prosody_scalogram_diff.py --today 0913 --output-root ../result
    python scripts/plot_prosody_scalogram_diff.py --show  # 保存に加えて画面表示もする場合
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

# プロット設定: (ファイル接頭辞, 凡例ラベル, 色)
SOURCES = [
    ("mtrf_pred", "predicted", "#2196F3"),
    ("mtrf_test", "raw EEG", "#E91E63"),
]


def _cwt(data, wavelet, widths, w):
    """
    scipy.signal.cwt の代替実装（scipy 1.12 で非推奨・1.15 で削除予定のため自前実装）．
    ロジックは旧 scipy.signal.cwt と同一（各 width でウェーブレットを生成し，
    信号とその共役反転版を 'same' モードで畳み込む）．

    Parameters
    ----------
    data    : 1-D array (n_samples,)
    wavelet : ウェーブレット生成関数（例: scipy.signal.morlet2）
    widths  : 各スケールに対応する width の配列
    w       : morlet2 に渡す形状パラメータ

    Returns
    -------
    output : 2-D array (len(widths), n_samples)  複素数のCWT係数
    """
    output = np.empty((len(widths),) + data.shape, dtype=np.complex128)
    for ind, width in enumerate(widths):
        N = int(np.min([10 * width, len(data)]))
        wavelet_data = np.conj(wavelet(N, width, w=w)[::-1])
        output[ind] = fftconvolve(data, wavelet_data, mode="same")
    return output


def compute_scalogram(signal, sr=1000, freqs=None):
    """
    CWT（連続ウェーブレット変換）によるスカログラムを計算する．

    Parameters
    ----------
    signal : 1-D array (n_samples,)
    sr     : サンプリングレート
    freqs  : 解析対象の周波数帯（Hz）

    Returns
    -------
    power_norm : 2-D array (n_freqs, n_samples)  正規化パワー
    """
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


def compute_diff_rms_per_layer(data_root, prefix, freqs):
    """
    指定した接頭辞（mtrf_pred / mtrf_test）について，
    全被験者・全単語平均のレイヤーごとスカログラム差分RMSを計算する．

    Returns
    -------
    diff_rms_per_layer : (N_LAYERS,) array もしくは None（有効データなしの場合）
    n_valid : 有効な (被験者, 単語) ペア数
    """
    diff_rms_per_layer = np.zeros(N_LAYERS)
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
                layer_diff_power = []
                for ch in range(N_CHANNELS):
                    sig_d = data_d[layer, ch, :]
                    sig_u = data_u[layer, ch, :]

                    scalo_d = compute_scalogram(sig_d, sr=EEG_SR, freqs=freqs)
                    scalo_u = compute_scalogram(sig_u, sr=EEG_SR, freqs=freqs)

                    diff = scalo_d - scalo_u
                    layer_diff_power.append(diff)

                mean_diff = np.mean(layer_diff_power, axis=0)
                rms = np.sqrt(np.mean(mean_diff ** 2))
                diff_rms_per_layer[layer] += rms

            n_valid += 1

    if n_valid == 0:
        return None, 0

    diff_rms_per_layer /= n_valid
    return diff_rms_per_layer, n_valid


def main():
    parser = argparse.ArgumentParser(
        description="韻律条件間（_d vs _u）のスカログラム差分RMSを，予測脳波・実測脳波を重ねてレイヤー毎にプロット")
    parser.add_argument("--data-root", type=str, default=None,
                        help="データディレクトリのルート（デフォルト: ./data/<TODAY>）")
    parser.add_argument("--today", type=str, default=TODAY,
                        help="日付ラベル（デフォルト: 0913）")
    parser.add_argument("--output-root", type=str, default="./result",
                        help="出力先ルートディレクトリ（デフォルト: ./result）。result/<today>/ に画像を保存する")
    parser.add_argument("--output", type=str, default=None,
                        help="出力画像ファイルパスを直接指定する場合（指定時は --output-root/--today より優先）")
    parser.add_argument("--show", action="store_true",
                        help="保存に加えて画面表示も行う")
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

    # ---- 予測(pred)・実測(test) の両方を計算 ----
    results = {}
    for prefix, label, color in SOURCES:
        print(f"\n--- {label} ({prefix}) を計算中 ---")
        diff_rms_per_layer, n_valid = compute_diff_rms_per_layer(data_root, prefix, freqs)
        if diff_rms_per_layer is None:
            print(f"警告: {prefix} の有効なデータが見つかりませんでした．このデータ系列はスキップします．")
            continue
        print(f"有効な (被験者, 単語) ペア数: {n_valid}")
        print(f"レイヤーごとの差分RMS: {diff_rms_per_layer}")
        results[prefix] = (diff_rms_per_layer, label, color)

    if not results:
        print("エラー: 予測・実測ともに有効なデータが見つかりませんでした．")
        return

    # ---- プロット（予測・実測を同一グラフに重ねる） ---->save_pngに移行
    plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    })

    
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = np.arange(N_LAYERS)

    for prefix, label, color in SOURCES:
        if prefix not in results:
            continue
        diff_rms_per_layer, label, color = results[prefix]
        ax.plot(layers, diff_rms_per_layer, "o-", color=color, linewidth=1.5,
                markersize=5, markerfacecolor="white", markeredgewidth=1.5,
                label=label)

    ax.set_xlabel("Whisper Layer")
    ax.set_ylabel("Scalogram Difference RMS\n(Normalized Power, H-L vs L-H)")
    # ax.set_title("Prosody Contrast (H-L vs L-H): Scalogram Power Difference per Layer\n(Predicted vs Measured EEG)",
                #  fontsize=14)
    ax.set_xticks(layers)
    ax.set_xticklabels(layers)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.legend()

    fig.tight_layout()

    # ---- 保存先の決定: result/<today>/ に自動保存 ----
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_root) / args.today
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "prosody_scalogram_diff_pred_vs_test.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n画像を保存しました: {output_path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()