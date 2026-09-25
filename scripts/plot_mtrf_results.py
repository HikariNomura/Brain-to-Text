# python ./scripts/plot_mtrf_results.py data/nocut --outdir result/nocut
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})

WORD_LIST = ["ame", "nizi", "momo", "hasi"]
OTO = ["_d", "_u"]

# チャンネル配置 (5行 x 6列)。空欄は None。
# データの "channel" 列は 0始まり (ch0〜ch20) なので、ここも0始まりで統一する。
GRID_LAYOUT = [
    [6,    7,  None, 16, None, None],
    [5,    8,    14, 17, None, None],
    [4,    9,    13, 18,    15,   0],
    [3,   10,    12, 19,    2,    1],
    [None, 11, None, 20, None, None],
]


CHANNEL_NAMES = [
    "C6",   # ch0
    "T8",   # ch1
    "TP8",  # ch2
    "F7",   # ch3
    "F5",   # ch4
    "F3",   # ch5
    "F1",   # ch6
    "FC1",  # ch7
    "FC3",  # ch8
    "FC5",  # ch9
    "FT7",  # ch10
    "FT9",  # ch11
    "T7",   # ch12
    "C5",   # ch13
    "C3",   # ch14
    "CP6",  # ch15
    "CP1",  # ch16
    "CP3",  # ch17
    "CP5",  # ch18
    "TP7",  # ch19
    "TP9",  # ch20
]

# 被験者ラベルに使うマーカー(丸・三角・四角、4人目以降はひし形などで追加)
SUBJECT_MARKERS = ["o", "^", "s", "D", "v", "P"]

PROSODY_COLORS = {"_d": "tab:blue", "_u": "tab:orange"}
PROSODY_NAMES = {"_d": "H-L", "_u": "L-H"}  # 日本語だと文字化けするため英語表記


def load_all(root_dir: Path) -> pd.DataFrame:
    """root_dir/被験者フォルダ/mtrf_agg_{word}{prosody}.csv を全部読み込み、
    subject/word/prosody列を付けて結合する。subjectは検出順にA, B, C...と
    匿名化したラベル(subject_label)も付与する。"""
    dfs = []
    # 隠しフォルダ(.で始まる)や、明らかにフォルダでないものを除外
    subject_dirs = [
        d for d in sorted(root_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]
    subject_label_map = {d.name: str(i + 1) for i, d in enumerate(subject_dirs)}    
    print(f"検出した被験者フォルダ(順番通り): {[d.name for d in subject_dirs]}")
    print(f"割り当てラベル: {subject_label_map}")

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        subject_label = f"S{subject_label_map[subject]}"
        for word in WORD_LIST:
            for prosody in OTO:
                path = subject_dir / f"mtrf_agg_{word}{prosody}.csv"
                if not path.exists():
                    print(f"[警告] 見つかりません: {path}")
                    continue
                df = pd.read_csv(path)
                df.insert(0, "prosody", prosody)
                df.insert(0, "word", word)
                df.insert(0, "subject_label", subject_label)
                df.insert(0, "subject", subject)
                dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"mtrf_agg_*.csv が1つも見つかりませんでした: {root_dir}")

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="mtrf_agg_{word}{prosody}.csv (layer,channel,mean_r,...) から"
                    "被験者×韻律のグラフを生成する"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="被験者フォルダ群が入っているディレクトリ (例: result/0905_v2_avgtest)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="画像の出力先ディレクトリ (省略時は root_dir と同じ場所)",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    outdir = Path(args.outdir) if args.outdir else root_dir
    outdir.mkdir(parents=True, exist_ok=True)

    df_all = load_all(root_dir)
    print(f"読み込み: {len(df_all)}行 "
          f"(subjects={df_all['subject_label'].unique().tolist()}, "
          f"words={df_all['word'].unique().tolist()}, "
          f"prosody={df_all['prosody'].unique().tolist()})")

    # =====================================================
    # 3. レイヤー平均の折れ線グラフ
    #    単語は平均 → 被験者3人×韻律(d/u)6本(薄め、被験者ごとにマーカー形状を変える)
    #    + 韻律ごとの被験者平均2本(太め) + 全体平均(一番太い黒)
    # =====================================================
    subj_prosody_mean = (
        df_all.groupby(["subject_label", "prosody", "layer"])["mean_r"]
        .mean()
        .reset_index()
        .sort_values(["subject_label", "prosody", "layer"])
    )

    prosody_mean = (
        subj_prosody_mean.groupby(["prosody", "layer"])["mean_r"]
        .mean()
        .reset_index()
        .sort_values(["prosody", "layer"])
    )

    overall_mean = (
        subj_prosody_mean.groupby("layer")["mean_r"]
        .mean()
        .reset_index()
        .sort_values("layer")
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    # 被験者ラベル -> マーカー の対応を固定
    subject_labels_sorted = sorted(df_all["subject_label"].unique())
    subject_marker_map = {
        label: SUBJECT_MARKERS[i % len(SUBJECT_MARKERS)]
        for i, label in enumerate(subject_labels_sorted)
    }

    # 個別被験者×韻律 (薄め、細め、被験者ごとにマーカー形状を変える)
    for (subject_label, prosody), g in subj_prosody_mean.groupby(["subject_label", "prosody"]):
        g = g.sort_values("layer")
        ax.plot(
            g["layer"],
            g["mean_r"],
            marker=subject_marker_map[subject_label],
            markersize=4,
            linewidth=1.0,
            alpha=0.3,
            color=PROSODY_COLORS[prosody],
            label=f"{subject_label} - {PROSODY_NAMES[prosody]}",
        )

    # 韻律ごとの被験者平均 (太め、はっきりした色)
    for prosody, g in prosody_mean.groupby("prosody"):
        g = g.sort_values("layer")
        ax.plot(
            g["layer"],
            g["mean_r"],
            marker="o",
            linewidth=3,
            alpha=0.95,
            color=PROSODY_COLORS[prosody],
            label=f"Mean {PROSODY_NAMES[prosody]}",
        )

    # 全体平均 (一番太い黒)
    ax.plot(
        overall_mean["layer"],
        overall_mean["mean_r"],
        marker="o",
        linewidth=3.5,
        color="black",
        label="Overall",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean r (word&channel-averaged)")
    # ax.set_title("Layer-wise mean r by subject & prosody")

    all_layers = sorted(df_all["layer"].unique())
    ax.set_xticks(all_layers)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        title="Series",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        # fontsize=8,
    )

    fig.tight_layout()
    fig_path = outdir / "layer_mean_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_path}")

    # =====================================================
    # 4. (参考) レイヤーごとのチャンネル平均を棒グラフで比較
    #    被験者・単語・韻律は全部平均
    # =====================================================
    channel_layer_all = (
        df_all.groupby(["layer", "channel"])["mean_r"]
        .mean()
        .reset_index()
    )

    layers = sorted(channel_layer_all["layer"].unique())

    pivot = channel_layer_all.pivot(
        index="channel",
        columns="layer",
        values="mean_r"
    ).sort_index()

    if len(layers) > 1:
        fig2, ax2 = plt.subplots(figsize=(14, 6))

        pivot.plot(kind="bar", ax=ax2)

        ax2.set_xlabel("Channel")
        ax2.set_ylabel("Mean r (subject/word/prosody averaged)")
        ax2.set_title("Channel-wise mean r by layer (all averaged)")

        ax2.set_xticklabels([str(ch + 1) for ch in pivot.index])

        ax2.legend(
            title="Layer",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            ncol=2,
        )

        ax2.grid(True, axis="y", alpha=0.3)

        fig2_path = outdir / "channel_mean_comparison_bar.png"
        fig2.tight_layout()
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        print(f"Saved: {fig2_path}")

    # =====================================================
    # 5. チャンネル配置図 (レイヤー0・31・全レイヤー平均のみを表示)
    #    被験者・単語・韻律は全部平均
    # =====================================================
    n_rows = len(GRID_LAYOUT)
    n_cols = len(GRID_LAYOUT[0])

    target_layers = [l for l in [0, 31] if l in pivot.columns]

    pivot_plot = pivot.copy()
    pivot_plot["Average"] = pivot.mean(axis=1)

    plot_layers = target_layers + ["Average"]

    vmin = pivot_plot[plot_layers].values.min()
    vmax = pivot_plot[plot_layers].values.max()

    n_panels = len(plot_layers)
    n_plot_cols = min(4, n_panels)
    n_plot_rows = math.ceil(n_panels / n_plot_cols)

    fig3, axes3 = plt.subplots(
        n_plot_rows,
        n_plot_cols,
        figsize=(4 * n_plot_cols, 3.5 * n_plot_rows),
        squeeze=False,
    )

    im = None

    for idx, layer in enumerate(plot_layers):
        r = idx // n_plot_cols
        c = idx % n_plot_cols
        ax = axes3[r][c]

        grid_values = np.full((n_rows, n_cols), np.nan)
        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None and ch in pivot_plot.index:
                    grid_values[i, j] = pivot_plot.loc[ch, layer]

        masked = np.ma.masked_invalid(grid_values)

        im = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax)

        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None:
                    val = grid_values[i, j]
                    # 表示上の番号は棒グラフ側 (ch + 1) の1始まり表記に合わせる
                    ax.text(
                        j, i,
                        f"{ch + 1}\n{val:.2f}",
                        ha="center", va="center",
                        fontsize=8, color="white",
                    )

        title = "Average (all layers)" if layer == "Average" else f"Layer {layer}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_panels, n_plot_rows * n_plot_cols):
        r = idx // n_plot_cols
        c = idx % n_plot_cols
        axes3[r][c].axis("off")

    fig3.colorbar(im, ax=axes3, shrink=0.7, label="Mean r")
    fig3.suptitle("Channel layout: Layer 0 / 31 / Average", fontsize=14)

    fig3_path = outdir / "channel_layout_comparison.png"
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved: {fig3_path}")

    # =====================================================
    # 6. チャンネル配置図 (全レイヤー平均のみ、部位名表示・数値なし)
    #    被験者・単語・韻律・レイヤーは全部平均
    # =====================================================
    avg_values = pivot_plot["Average"].values
    vmin6 = avg_values.min()
    vmax6 = avg_values.max()
 
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
 
    grid_values6 = np.full((n_rows, n_cols), np.nan)
    for i, row in enumerate(GRID_LAYOUT):
        for j, ch in enumerate(row):
            if ch is not None and ch in pivot_plot.index:
                grid_values6[i, j] = pivot_plot.loc[ch, "Average"]
 
    masked6 = np.ma.masked_invalid(grid_values6)
    im4 = ax4.imshow(masked6, cmap="Blues", vmin=vmin6, vmax=vmax6)

    cmap6 = plt.get_cmap("Blues")
    norm6 = plt.Normalize(vmin=vmin6, vmax=vmax6)
 
    for i, row in enumerate(GRID_LAYOUT):
        for j, ch in enumerate(row):
            if ch is not None:
                name = CHANNEL_NAMES[ch] if 0 <= ch < len(CHANNEL_NAMES) else f"ch{ch}"
                # r, g, b, _ = cmap6(norm6(val))
                # luminance = 0.299 * r + 0.587 * g + 0.114 * b
                # text_color =  "#ffffff" if luminance > 0.39 else "#000000"
                text_color = "#ffffff" if grid_values6[i,j] > 0.2 else "#000000"
                val = grid_values6[i,j]
                ax4.text(
                    j, i,
                    f"{name}\n{val:.2f}",
                    ha="center", va="center",
                    fontsize=12, color= text_color
                )
 
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.axvline(x=3.5, linewidth=2, color="k")
    fig4.colorbar(im4, ax=ax4, shrink=0.8, label="Correlation (r)" )
 
    fig4_path = outdir / "channel_layout_average.png"
    fig4.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close(fig4)
 
    print(f"Saved: {fig4_path}")


    # =====================================================
    # 7. チャンネル配置図 (Layer 20 - Layer 0 の差分を1枚に、部位名表示・数値なし)
    #    被験者・単語・韻律は全部平均
    # =====================================================
    if 0 in pivot.columns and 20 in pivot.columns:
        diff_series = pivot[20] - pivot[0]
 
        diff_abs_max = diff_series.abs().max()
        vmin7, vmax7 = -diff_abs_max, diff_abs_max
 
        grid_values7 = np.full((n_rows, n_cols), np.nan)
        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None and ch in diff_series.index:
                    grid_values7[i, j] = diff_series.loc[ch]
 
        masked7 = np.ma.masked_invalid(grid_values7)
 
        fig5, ax5 = plt.subplots(figsize=(5.5, 3.5))
        im5 = ax5.imshow(masked7, cmap="coolwarm", vmin=vmin7, vmax=vmax7)
 
        cmap7 = plt.get_cmap("coolwarm")
        norm7 = plt.Normalize(vmin=vmin7, vmax=vmax7)
 
        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None:
                    name = CHANNEL_NAMES[ch] if 0 <= ch < len(CHANNEL_NAMES) else f"ch{ch}"
                    val = grid_values7[i, j]
                    if np.isnan(val):
                        text_color = "#000000"
                    else:
                        r, g, b, _ = cmap7(norm7(val))
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        text_color = "#ffffff" if luminance < 0.60 else "#000000"
                    ax5.text(
                        j, i,
                        f"{name}\n{val:.2f}",
                        ha="center", va="center",
                        fontsize=12, color=text_color,
                    )
 
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.axvline(x=3.5, linewidth=2, color="k")
        fig5.colorbar(im5, ax=ax5, shrink=0.8, label="Δr (Layer 20 − 0)")
 
        fig5_path = outdir / "channel_layout_layer20_minus_layer0.png"
        fig5.savefig(fig5_path, dpi=150, bbox_inches="tight")
        plt.close(fig5)
 
        print(f"Saved: {fig5_path}")


if __name__ == "__main__":
    main()