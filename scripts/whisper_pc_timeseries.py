import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


WORD_LIST = ["ame", "nizi", "momo", "hasi"]
OTO = ["_d", "_u"]
AUDIO_LENGTH_SEC = 3.0
WHISPER_SR = 50

N_PCA_SHOW = 2   # 表示する主成分数(第1・第2)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
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


def main():
    parser = argparse.ArgumentParser(
        description="Whisper特徴量のPCA後、第1・第2主成分を時系列で可視化し、"
                    "d条件とu条件を単語ごとに重ねて表示する"
    )
    parser.add_argument("whisper_root", type=str)
    parser.add_argument("--layer", type=int, default=20, help="解析対象のレイヤー")
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    whisper_root = Path(args.whisper_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    layer = args.layer
    time_axis = np.arange(int(AUDIO_LENGTH_SEC * WHISPER_SR)) / WHISPER_SR

    # 全単語・両条件をまとめてPCAを1つ学習(共通の軸で比較するため)
    all_feats = []
    per_word_features = {}
    for word in WORD_LIST:
        feat_d = load_feature(whisper_root, word, OTO[0], layer)
        feat_u = load_feature(whisper_root, word, OTO[1], layer)
        per_word_features[word] = (feat_d, feat_u)
        all_feats.append(feat_d)
        all_feats.append(feat_u)

    combined = np.vstack(all_feats)
    pca = PCA(n_components=N_PCA_SHOW).fit(combined)
    explained = pca.explained_variance_ratio_

    print(f"Layer {layer}: PC1 寄与率 = {explained[0]*100:.2f}%, "
          f"PC2 寄与率 = {explained[1]*100:.2f}%")

    # =====================================================
    # 図: 4単語(行) × 2主成分(列) のグリッド、各パネルにd/uを重ねる
    # =====================================================
    fig, axes = plt.subplots(len(WORD_LIST), N_PCA_SHOW,
                              figsize=(6 * N_PCA_SHOW, 3 * len(WORD_LIST)),
                              sharex=True)

    for row, word in enumerate(WORD_LIST):
        feat_d, feat_u = per_word_features[word]
        pcs_d = pca.transform(feat_d)
        pcs_u = pca.transform(feat_u)

        for col in range(N_PCA_SHOW):
            ax = axes[row][col]
            ax.plot(time_axis, pcs_d[:, col], linewidth=2, color="tab:blue", label="d (accented)")
            ax.plot(time_axis, pcs_u[:, col], linewidth=2, color="tab:orange", label="u (unaccented)")

            if row == 0:
                ax.set_title(f"PC{col+1} ({explained[col]*100:.1f}% var.)")
            if col == 0:
                ax.set_ylabel(f"{word}\nvalue")
            if row == len(WORD_LIST) - 1:
                ax.set_xlabel("Time (s)")

            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=10)

    fig.suptitle(f"Layer {layer}: PCA components over time (d vs u, per word)", fontsize=17)
    fig.tight_layout()

    fig_path = outdir / f"whisper_pc_timeseries_layer{layer}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # =====================================================
    # 追加: 単語をまたいでPC1・PC2を重ね書きした図(比較しやすくするため)
    # =====================================================
    fig2, axes2 = plt.subplots(1, N_PCA_SHOW, figsize=(7 * N_PCA_SHOW, 5))

    word_linestyles = ["-", "--", "-.", ":"]

    for col in range(N_PCA_SHOW):
        ax = axes2[col]
        for word, ls in zip(WORD_LIST, word_linestyles):
            feat_d, feat_u = per_word_features[word]
            pcs_d = pca.transform(feat_d)
            pcs_u = pca.transform(feat_u)

            ax.plot(time_axis, pcs_d[:, col], linewidth=1.8, linestyle=ls,
                    color="tab:blue", alpha=0.8, label=f"{word} (d)")
            ax.plot(time_axis, pcs_u[:, col], linewidth=1.8, linestyle=ls,
                    color="tab:orange", alpha=0.8, label=f"{word} (u)")

        ax.set_title(f"PC{col+1} ({explained[col]*100:.1f}% var.)")
        ax.set_xlabel("Time (s)")
        if col == 0:
            ax.set_ylabel("Component value")
        ax.grid(True, alpha=0.3)

    axes2[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig2.suptitle(f"Layer {layer}: PC1/PC2 overlaid across words (d vs u)", fontsize=16)
    fig2.tight_layout()

    fig2_path = outdir / f"whisper_pc_overlay_layer{layer}.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")


if __name__ == "__main__":
    main()