import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "figure.titlesize": 22,
})

# チャンネル配置 (5行 x 6列)。空欄は None。
# データの "channel" 列は 0始まり (ch0〜ch20) なので、ここも0始まりで統一する。
GRID_LAYOUT = [
    [6,    7,  None, 16, None, None],
    [5,    8,    14, 17, None, None],
    [4,    9,    13, 18,    0,   15],
    [3,   10,    12, 19,    1,    2],
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

# 集計対象のファイル名パターン (layer,channel,mean_r,... を持つCSV)
AGG_PATTERN = "mtrf_agg_*.csv"

CONDITION_MARKERS = ["o", "^", "s", "D", "v", "P"]

CONDITION_COLORS = [
    "#FF4B00",
    "#005AFF",
    "#03AF7A",
    "#4DC4FF",
    "#F6AA00",
    "#FFF100",
]

def find_conditions(root_dirs):
    """複数のroot_dir直下から、配下(何階層でもよい)にmtrf_agg_*.csvを
    持つフォルダを条件として検出。
    内部キーは '日付フォルダ名/条件フォルダ名' で一意にし、
    表示名は条件フォルダ名のみ（同名が複数日付にある場合だけ日付を付けて区別）にする。"""
    entries = []  # (internal_key, root_name, cond_name, cond_dir)
    for root_dir in root_dirs:
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir():
                continue
            agg_files = list(d.rglob(AGG_PATTERN))
            if agg_files:
                internal_key = f"{root_dir.name}/{d.name}"
                entries.append((internal_key, root_dir.name, d.name, d))

    name_counts = Counter(cond_name for _, _, cond_name, _ in entries)

    conditions = {}       # internal_key -> condition_dir
    display_labels = {}   # internal_key -> 表示名
    for internal_key, root_name, cond_name, cond_dir in entries:
        conditions[internal_key] = cond_dir
        if name_counts[cond_name] > 1:
            display_labels[internal_key] = f"{cond_name} ({root_name})"
        else:
            display_labels[internal_key] = cond_name

    return conditions, display_labels


def load_condition_data(cond_dir: Path) -> pd.DataFrame:
    """条件フォルダ配下の全 mtrf_agg_*.csv (被験者×単語×韻律ぶん)を結合して返す。
    各ファイルは layer, channel, mean_r, ... の列を持つ。"""
    files = sorted(cond_dir.rglob(AGG_PATTERN))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file"] = str(f.relative_to(cond_dir))
        dfs.append(df)
    if not dfs:
        raise ValueError(f"{cond_dir} 配下に {AGG_PATTERN} が見つかりません")
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="複数の日付フォルダ・複数条件の mtrf_agg_*.csv "
                    "(layer,channel,mean_r,...) を集めて比較グラフを作る"
    )
    parser.add_argument(
        "root_dirs",
        type=str,
        nargs="+",
        help="条件フォルダ群が入っているルートディレクトリ (複数指定可、例: result/0831 result/0901)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="出力先ディレクトリ (省略時は先頭のroot_dirの親/comparison)",
    )
    args = parser.parse_args()

    root_dirs = [Path(r) for r in args.root_dirs]
    for r in root_dirs:
        if not r.exists():
            raise FileNotFoundError(f"見つかりません: {r}")

    outdir = Path(args.outdir) if args.outdir else root_dirs[0].parent / "comparison"
    outdir.mkdir(parents=True, exist_ok=True)

    conditions, display_labels = find_conditions(root_dirs)
    if not conditions:
        raise ValueError(f"{AGG_PATTERN} を持つ条件フォルダが見つかりません: {root_dirs}")

    print(f"検出した条件: {[display_labels[k] for k in conditions]}")

    # =====================================================
    # データ読み込み・集計
    #   layer_avg[label]  : layerごとに channel/被験者/単語/韻律 を平均
    #   channel_avg[label]: channelごとに layer/被験者/単語/韻律 を平均
    # 以降はすべて「表示名(label)」をキーにして扱う
    # =====================================================
    layer_avg = {}    # label -> DataFrame(layer, mean_r)
    channel_avg = {}  # label -> DataFrame(channel, mean_r)

    for internal_key, cond_dir in conditions.items():
        label = display_labels[internal_key]

        df_all = load_condition_data(cond_dir)
        print(f"  {label}: {df_all['__source_file'].nunique()}ファイル, {len(df_all)}行")

        layer_avg[label] = (
            df_all.groupby("layer")["mean_r"]
            .mean()
            .reset_index()
            .rename(columns={"mean_r": "channel_mean_r"})  # 以降の描画コードとの互換のため列名を揃える
            .sort_values("layer")
        )

        channel_avg[label] = (
            df_all.groupby("channel")["mean_r"]
            .mean()
            .reset_index()
            .rename(columns={"mean_r": "fold_mean_r"})  # 以降の描画コードとの互換のため列名を揃える
            .sort_values("channel")
        )

    # =====================================================
    # 1. layer_mean_comparison.png
    #    条件ごとの折れ線を1枚に重ねる (x=layer)
    # =====================================================
    fig, ax = plt.subplots(figsize=(17, 6))

    all_layers = sorted({l for df in layer_avg.values() for l in df["layer"]})

    for i, (label, df) in enumerate(layer_avg.items()):

        marker = CONDITION_MARKERS[
            i % len(CONDITION_MARKERS)
        ]

        color = CONDITION_COLORS[
            i % len(CONDITION_COLORS)
        ]

        ax.plot(
            df["layer"],
            df["channel_mean_r"],
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.5,
            color=color,
            linewidth=2,
            markersize=7,
            label=label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation (r)")
    # ax.set_title("Layer-wise mean r by condition")
    ax.set_xticks(all_layers)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        # title="Condition",
        loc="upper right",
        bbox_to_anchor=(1.25, 1),
        borderaxespad=1
        # ncol = 3
    )
    ax.set_ylim(0.03,0.18)

    fig.tight_layout()
    fig_path = outdir / "layer_mean_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_path}")

    # # =====================================================
    # # 2. channel_mean_comparison_bar.png
    # #    同じデータ(全チャンネル平均)を条件比較の棒グラフで (x=layer, 系列=条件)
    # # =====================================================
    # layer_pivot = pd.DataFrame({label: df.set_index("layer")["channel_mean_r"] for label, df in layer_avg.items()})
    # layer_pivot = layer_pivot.sort_index()

    # fig2, ax2 = plt.subplots(figsize=(14, 6))

    # layer_pivot.plot(kind="bar", ax=ax2)

    # ax2.set_xlabel("Layer")
    # ax2.set_ylabel("Mean r (averaged over channel/subject/word/prosody)")
    # ax2.set_title("Layer-wise mean r by condition (channel-averaged)")
    # ax2.legend(
    #     title="Condition",
    #     loc="upper left",
    #     bbox_to_anchor=(1.01, 1.0),
    #     borderaxespad=0,
    #     fontsize=8,
    # )
    # ax2.grid(True, axis="y", alpha=0.3)

    # fig2_path = outdir / "channel_mean_comparison_bar.png"
    # fig2.tight_layout()
    # fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    # plt.close(fig2)

    # print(f"Saved: {fig2_path}")

    # =====================================================
    # 3. channel_layout_comparison.png
    #    条件ごとに1パネル、各チャンネルは全レイヤー平均値で色分け
    # =====================================================
    n_rows = len(GRID_LAYOUT)
    n_cols = len(GRID_LAYOUT[0])

    all_vals = np.concatenate([df["fold_mean_r"].values for df in channel_avg.values()])
    vmin = all_vals.min()
    vmax = all_vals.max()

    label_list = list(channel_avg.keys())
    n_panels = len(label_list)
    n_plot_cols = min(3, n_panels)
    n_plot_rows = math.ceil(n_panels / n_plot_cols)

    fig3, axes3 = plt.subplots(
        n_plot_rows,
        n_plot_cols,
        figsize=(4.5 * n_plot_cols, 4 * n_plot_rows),
        squeeze=False,
    )

    im = None

    for idx, label in enumerate(label_list):
        r = idx // n_plot_cols
        c = idx % n_plot_cols
        ax = axes3[r][c]

        df = channel_avg[label].set_index("channel")["fold_mean_r"]

        grid_values = np.full((n_rows, n_cols), np.nan)
        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None and ch in df.index:
                    grid_values[i, j] = df.loc[ch]

        masked = np.ma.masked_invalid(grid_values)

        im = ax.imshow(masked, cmap="Blues", vmin=vmin, vmax=vmax)

        for i, row in enumerate(GRID_LAYOUT):
            for j, ch in enumerate(row):
                if ch is not None:
                    val = grid_values[i, j]
                    ax.text(j, i, CHANNEL_NAMES[ch],
                        ha="center", va="center",
                        fontsize=12, color="#ffffff" if grid_values[i,j]>0.20 else "#000000",
                    )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_panels, n_plot_rows * n_plot_cols):
        r = idx // n_plot_cols
        c = idx % n_plot_cols
        axes3[r][c].axis("off")

    fig3.colorbar(im, ax=axes3, shrink=0.7, label="Correlation (r)")
    # fig3.suptitle("Channel layout by condition (all-layer averaged)")

    fig3_path = outdir / "channel_layout_comparison.png"
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved: {fig3_path}")


if __name__ == "__main__":
    main()