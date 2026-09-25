# チャンネルごとに、レイヤーが深くなると相関がどうなるかみるやつ。
# 縦相関、横層、21本でカラフル

import argparse
import colorsys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 11,
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

# 対象とする条件フォルダ名 (これだけを使う)
TARGET_CONDITION_NAME = "0.5~08Hz"


def find_target_condition_dirs(root_dirs, target_name=TARGET_CONDITION_NAME):
    """複数のroot_dir直下から、フォルダ名が target_name に一致し、
    配下(何階層でもよい)に mtrf_agg_*.csv を持つフォルダを探す。
    複数のroot_dirにまたがって見つかった場合は、そのデータをすべて結合する。"""
    matched_dirs = []
    for root_dir in root_dirs:
        for d in sorted(root_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name != target_name:
                continue
            agg_files = list(d.rglob(AGG_PATTERN))
            if agg_files:
                matched_dirs.append(d)

    return matched_dirs


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


def channel_label(ch: int) -> str:
    if 0 <= ch < len(CHANNEL_NAMES):
        return f"{CHANNEL_NAMES[ch]} (ch{ch+1})"
    return f"ch{ch}"


def build_channel_colors(channels):
    """チャンネルごとに色を割り当てる。並び順(色相)はチャンネル番号順のまま
    (＝番号が近いチャンネルは色相も近い)にしつつ、明度・彩度を短い周期で
    変化させることで、色相が近い隣り合うチャンネル同士でも見分けがつく
    ようにする。"""
    n = len(channels)
    value_cycle = [1.0, 0.55, 0.8, 0.4]   # 明度を短い周期で変化
    sat_cycle = [1.0, 0.6, 0.85]          # 彩度も別の周期で変化 (明度周期とずらす)

    colors = {}
    for i, ch in enumerate(channels):
        hue = i / n
        value = value_cycle[i % len(value_cycle)]
        sat = sat_cycle[i % len(sat_cycle)]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, value)
        colors[ch] = (r, g, b, 1.0)
    return colors


def is_dark(rgba) -> bool:
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 0.55


def main():
    parser = argparse.ArgumentParser(
        description=f"'{TARGET_CONDITION_NAME}' 条件の mtrf_agg_*.csv (layer,channel,mean_r,...) を集めて"
                    "チャンネルごとの層別相関係数の推移グラフを作る"
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

    target_dirs = find_target_condition_dirs(root_dirs)
    if not target_dirs:
        raise ValueError(
            f"'{TARGET_CONDITION_NAME}' という名前の条件フォルダ ({AGG_PATTERN} を含む) が"
            f"見つかりません: {root_dirs}"
        )

    print(f"対象フォルダ: {[str(d) for d in target_dirs]}")

    # =====================================================
    # データ読み込み・集計
    #   channel_layer_avg: (channel, layer)ごとに 被験者/単語/韻律 を平均
    # =====================================================
    df_all = pd.concat([load_condition_data(d) for d in target_dirs], ignore_index=True)
    print(f"  {TARGET_CONDITION_NAME}: {df_all['__source_file'].nunique()}ファイル, {len(df_all)}行")

    channel_layer_avg = (
        df_all.groupby(["channel", "layer"])["mean_r"]
        .mean()
        .reset_index()
        .sort_values(["channel", "layer"])
    )

    channels = sorted(channel_layer_avg["channel"].unique())
    all_layers = sorted(channel_layer_avg["layer"].unique())

    # =====================================================
    # 1. channel_layer_lines.png
    #    チャンネルごとの折れ線を1枚に重ねる (x=layer, y=mean_r)
    # =====================================================
    fig, ax = plt.subplots(figsize=(13, 7))

    channel_colors = build_channel_colors(channels)

    for ch in channels:
        df_ch = channel_layer_avg[channel_layer_avg["channel"] == ch].sort_values("layer")
        ax.plot(
            df_ch["layer"],
            df_ch["mean_r"],
            marker="o",
            markeredgewidth=0,
            color=channel_colors[ch],
            linewidth=1.8,
            markersize=5,
            label=channel_label(ch),
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation (r)")
    ax.set_title(f"Layer-wise mean r by channel ({TARGET_CONDITION_NAME})")
    ax.set_xticks(all_layers)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        title="Channel",
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
        borderaxespad=0,
        ncol=1,
    )

    fig.tight_layout()
    fig_path = outdir / f"channel_layer_lines{TARGET_CONDITION_NAME}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_path}")

    # =====================================================
    # 2. channel_layout_all_frequency.png
    #    チャンネル空間マップ (all_frequency 1枚のみ)
    #    色は相関値のグラデーションではなく、1枚目の折れ線と同じ
    #    チャンネルごとの色をそのまま四角の色として使う
    # =====================================================
    n_rows = len(GRID_LAYOUT)
    n_cols = len(GRID_LAYOUT[0])

    fig2, ax2 = plt.subplots(figsize=(5, 4.5))

    grid_rgba = np.ones((n_rows, n_cols, 4))  # デフォルトは白 (データなしセル)
    for i, row in enumerate(GRID_LAYOUT):
        for j, ch in enumerate(row):
            if ch is not None and ch in channel_colors:
                grid_rgba[i, j] = channel_colors[ch]

    ax2.imshow(grid_rgba)

    for i, row in enumerate(GRID_LAYOUT):
        for j, ch in enumerate(row):
            if ch is not None:
                if ch in channel_colors:
                    text_color = "#ffffff" if is_dark(channel_colors[ch]) else "#000000"
                else:
                    # データが無いチャンネル (マスは白のまま) はラベルを黒字で表示
                    text_color = "#000000"
                ax2.text(
                    j, i, CHANNEL_NAMES[ch],
                    ha="center", va="center",
                    fontsize=12, color=text_color,
                )

    ax2.set_title(TARGET_CONDITION_NAME)
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig2_path = outdir / f"channel_layout_{TARGET_CONDITION_NAME}.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: {fig2_path}")


if __name__ == "__main__":
    main()