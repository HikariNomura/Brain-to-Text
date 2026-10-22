#!/usr/bin/env python3
"""
発話音声 × 発話時EEG の mTRF (Ridge) 解析。

方針(案1): Whisper特徴量をディスクに保存せず、試行ごとにオンザフライで抽出する。
  Pass 1 : 先頭 n_pca 試行で層ごとの IncrementalPCA(10次元固定) をフィット
  Pass 2 : 全試行について
             npyから音声(最後の列)とEEGを読む -> Whisper推論 -> 全層をfloat16で保持
             -> 層ごとにPCA射影 -> EEGのfsへ補間 -> ラグ展開(試行内ゼロパディング)
             -> 試行ごとの統計量 (n, Σx, Σy, Σy², X^T X, X^T Y) を計算 -> 特徴量は破棄
  その後  : 統計量の足し引きだけで fold ごとの Ridge (λは tune で選択) を解き、
             test 試行ごとの r / R^2 を出して平均 -> CSV

分割: train80 / tune5 / test15 (割合 80:5:15)。試行数Nが100でないときは
      test=round(N*0.15), tune=round(N*0.05), 残りをtrain とし、foldごとに
      train/tune/test が重ならないようにする。各試行は必ずちょうど1回testになる。

出力CSVの列: layer, channel, fold, train_n, tune_n, test_n, mean_r, mean_r2
  (train_n / tune_n / test_n は「試行数」)

入力: --data-dir 内の *.npy (ソート順 = 試行順)
  shape = (n_samples, n_cols)。先頭 n_eeg_ch(=21) 列がEEG、最後の列(=24番目)が発話音声。
  --npy-fs : npyのサンプリング周波数(音声もこのfs)
  --eeg-fs : 解析に使うEEGのfs (省略時は npy-fs。違う場合はEEGだけリサンプル)
  .\scripts\speech_online.py --data-dir npy/ --npy-fs 1000 --eeg-fs 1000 --out-csv data/mtrf_80_5_15_xxx.csv 
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WHISPER_FPS = 50.0  # Whisperエンコーダ出力のフレームレート(20ms/frame)
N_COMP = 10         # PCA次元(固定)


# ---------------------------------------------------------------- 特徴量抽出
class WhisperExtractor:
    """16kHz波形(1次元) -> (n_layers, T_frames, 1280) float16 (numpy)。層0..n_layers-1 = hidden_states[1:]"""

    def __init__(self, model_name="openai/whisper-large-v3", device=None):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = (
            WhisperModel.from_pretrained(model_name, torch_dtype=self.dtype)
            .encoder.to(self.device)
            .eval()
        )

    def __call__(self, wav):
        torch = self.torch
        dur = len(wav) / 16000
        assert dur <= 30.0, "30秒超の音声は未対応"
        n_frames = int(np.ceil(dur * WHISPER_FPS))
        inp = self.fe(wav, sampling_rate=16000, return_tensors="pt").input_features
        inp = inp.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.encoder(inp, output_hidden_states=True)
            hs = torch.stack(out.hidden_states[1:], 0)[:, 0, :n_frames]  # (L, T, 1280)
        return hs.to(torch.float16).cpu().numpy()


# ---------------------------------------------------------------- 読み込み
def resample(x, fs_in, fs_out, axis=0):
    from fractions import Fraction

    from scipy.signal import resample_poly

    if abs(fs_in - fs_out) < 1e-9:
        return x
    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=axis)


def load_trial(path, npy_fs, eeg_fs, n_eeg_ch, audio_col=-1):
    """npy -> (audio@16kHz float32 1次元, eeg@eeg_fs (n, n_eeg_ch))"""
    d = np.load(path)
    assert d.ndim == 2 and d.shape[1] > n_eeg_ch, f"{path}: shape={d.shape}"
    audio = d[:, audio_col].astype(np.float64)
    audio = audio - audio.mean()
    audio = audio / (np.abs(audio).max() + 1e-12)
    audio = resample(audio, npy_fs, 16000).astype(np.float32)
    eeg = resample(d[:, :n_eeg_ch].astype(np.float64), npy_fs, eeg_fs, axis=0)
    return audio, eeg


# ---------------------------------------------------------------- 前処理
def make_lagged(x, lags):
    """x: (T, F) -> (T, len(lags)*F)。out[t] = x[t - lag]。範囲外は0(試行内で完結)。"""
    T, F = x.shape
    out = np.zeros((T, len(lags) * F), dtype=np.float32)
    for i, lag in enumerate(lags):
        sl = slice(i * F, (i + 1) * F)
        if lag >= 0:
            if lag < T:
                out[lag:, sl] = x[: T - lag]
        else:
            if -lag < T:
                out[: T + lag, sl] = x[-lag:]
    return out


def lag_samples(fs, lag_min_ms, lag_max_ms):
    lo = int(np.ceil(lag_min_ms * fs / 1000.0))
    hi = int(np.floor(lag_max_ms * fs / 1000.0))
    return np.arange(lo, hi + 1)


def to_eeg_time(z, n_eeg, fs):
    """z: (T_frames, F) @50Hz -> (n_eeg, F) @fs (線形補間)"""
    t_feat = np.arange(z.shape[0]) / WHISPER_FPS
    t_eeg = np.arange(n_eeg) / fs
    return np.stack([np.interp(t_eeg, t_feat, z[:, j]) for j in range(z.shape[1])], 1)


# ---------------------------------------------------------------- 統計量
def trial_stats(X, Y):
    """1試行の十分統計量。X:(T,D) Y:(T,C)"""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    return dict(
        n=X.shape[0],
        sx=X.sum(0),
        sy=Y.sum(0),
        syy=(Y**2).sum(0),
        XtX=X.T @ X,
        XtY=X.T @ Y,
    )


def trial_metrics(S, layer, i, W, mx, my):
    """統計量だけで試行iのチャンネル別 r, R^2 を計算。
    予測 yhat = (x - mx) W + my"""
    n = S["n"][i]
    sx, sy, syy = S["sx"][layer][i], S["sy"][i], S["syy"][i]
    XtX = S["XtX"][layer][i].astype(np.float64)
    XtY = S["XtY"][layer][i].astype(np.float64)

    m = (sx / n - mx) @ W                                   # mean(yhat')
    XcX = XtX - np.outer(mx, sx) - np.outer(sx, mx) + n * np.outer(mx, mx)
    XcY = XtY - np.outer(mx, sy)
    syp2 = np.einsum("dc,dc->c", W, XcX @ W)                # Σ yhat'^2
    sypy = np.einsum("dc,dc->c", W, XcY)                    # Σ yhat' y
    ybar = sy / n

    var_p = syp2 / n - m**2
    cov = sypy / n - m * ybar
    var_y = syy / n - ybar**2
    r = cov / np.sqrt(np.clip(var_p * var_y, 1e-30, None))

    sse = syy - 2 * (sypy + my * sy) + (syp2 + 2 * my * n * m + n * my**2)
    sst = syy - n * ybar**2
    r2 = 1.0 - sse / np.clip(sst, 1e-30, None)
    return r, r2


def make_folds(n_trials, test_frac=0.15, tune_frac=0.05):
    """train:tune:test = 80:5:15。foldごとに3集合は互いに重ならない。
    testはarray_splitで分割するので全試行がちょうど1回testになる(サイズは±1)。
    tuneはtestの直後(循環)の試行から取り、残りがtrain。"""
    n_te = max(1, int(round(n_trials * test_frac)))
    n_tu = max(1, int(round(n_trials * tune_frac)))
    if n_te + n_tu >= n_trials:
        raise ValueError(f"試行数{n_trials}が少なすぎて分割できません")
    K = max(2, int(round(n_trials / n_te)))
    folds = []
    for k, te in enumerate(np.array_split(np.arange(n_trials), K)):
        order = (te[-1] + 1 + np.arange(n_trials)) % n_trials
        order = order[~np.isin(order, te)]
        folds.append((k, np.sort(order[n_tu:]), np.sort(order[:n_tu]), te))
    return folds


def evaluate_layer(S, layer, folds, lambdas_rel):
    rows = []
    n, sy = S["n"], S["sy"]
    sx, XtX, XtY = S["sx"][layer], S["XtX"][layer], S["XtY"][layer]
    n_ch = sy.shape[1]
    for k, tr, tu, te in folds:
        ntr = n[tr].sum()
        mx = sx[tr].sum(0) / ntr
        my = sy[tr].sum(0) / ntr
        A = XtX[tr].sum(0, dtype=np.float64) - ntr * np.outer(mx, mx)
        B = XtY[tr].sum(0, dtype=np.float64) - ntr * np.outer(mx, my)
        ev, Q = np.linalg.eigh(A)
        ev = np.clip(ev, 0, None)
        QtB = Q.T @ B
        scale = ev.mean()

        Ws = [Q @ (QtB / (ev[:, None] + lam * scale)) for lam in lambdas_rel]
        tune_scores = [
            np.nanmean([trial_metrics(S, layer, i, W, mx, my)[0].mean() for i in tu])
            for W in Ws
        ]
        W = Ws[int(np.nanargmax(tune_scores))]

        res = [trial_metrics(S, layer, i, W, mx, my) for i in te]
        r = np.nanmean([x[0] for x in res], 0)
        r2 = np.nanmean([x[1] for x in res], 0)
        for ch in range(n_ch):
            rows.append(
                dict(layer=layer, channel=ch, fold=k, train_n=len(tr), tune_n=len(tu),
                     test_n=len(te), mean_r=r[ch], mean_r2=r2[ch])
            )
    return rows


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="試行ごとの*.npyのあるフォルダ")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--npy-fs", type=float, required=True, help="npyのサンプリング周波数[Hz]")
    ap.add_argument("--eeg-fs", type=float, default=None, help="解析用EEG fs (省略時=npy-fs)")
    ap.add_argument("--n-eeg-ch", type=int, default=21, help="先頭からEEGとして使う列数")
    ap.add_argument("--lag-min-ms", type=float, default=1)
    ap.add_argument("--lag-max-ms", type=float, default=200)
    ap.add_argument("--n-pca-trials", type=int, default=20, help="PCAフィットに使う試行数")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--lambdas-rel", type=float, nargs="+",
                    default=[10.0**k for k in range(-4, 5)],
                    help="λ候補(X^T Xの固有値平均に対する相対値)")
    args = ap.parse_args()
    eeg_fs = args.eeg_fs or args.npy_fs

    from sklearn.decomposition import IncrementalPCA

    files = sorted(Path(args.data_dir).glob("*.npy"))
    assert files, "npyが見つかりません"
    N = len(files)
    lags = lag_samples(eeg_fs, args.lag_min_ms, args.lag_max_ms)
    folds = make_folds(N)
    k0, tr0, tu0, te0 = folds[0]
    print(f"trials={N}, folds={len(folds)} (例 fold0: train{len(tr0)}/tune{len(tu0)}/test{len(te0)}), "
          f"lags={lags[0]}..{lags[-1]} samples ({len(lags)}個)")
    if args.npy_fs < 8000:
        print(f"[警告] npy-fs={args.npy_fs}Hz では音声の帯域が狭く(上限{args.npy_fs / 2}Hz)、"
              "Whisper特徴量の質が落ちる可能性があります")

    ext = WhisperExtractor(args.model)

    # ---- Pass 1: 層ごとのPCA
    ipcas = None
    n_fit = min(args.n_pca_trials, N)
    for i in range(n_fit):
        audio, _ = load_trial(files[i], args.npy_fs, eeg_fs, args.n_eeg_ch)
        hs = ext(audio)
        if ipcas is None:
            ipcas = [IncrementalPCA(n_components=N_COMP, whiten=True) for _ in range(hs.shape[0])]
        for l in range(hs.shape[0]):
            ipcas[l].partial_fit(hs[l].astype(np.float32))
        print(f"[PCA fit] {i + 1}/{n_fit}", flush=True)
    L = len(ipcas)

    # ---- Pass 2: 試行ごとの統計量
    D = N_COMP * len(lags)
    C = args.n_eeg_ch
    S = dict(
        n=np.zeros(N), sy=np.zeros((N, C)), syy=np.zeros((N, C)),
        sx=np.zeros((L, N, D)),
        XtX=np.zeros((L, N, D, D), dtype=np.float32),
        XtY=np.zeros((L, N, D, C), dtype=np.float32),
    )
    for i in range(N):
        audio, eeg = load_trial(files[i], args.npy_fs, eeg_fs, args.n_eeg_ch)
        hs = ext(audio)                                    # (L, T, 1280) float16
        n_eeg = min(len(eeg), int(hs.shape[1] / WHISPER_FPS * eeg_fs))
        Y = eeg[:n_eeg]
        for l in range(L):
            z = ipcas[l].transform(hs[l].astype(np.float32))          # (T, N_COMP)
            X = make_lagged(to_eeg_time(z, n_eeg, eeg_fs), lags)       # (n_eeg, D)
            st = trial_stats(X, Y)
            S["sx"][l, i] = st["sx"]
            S["XtX"][l, i] = st["XtX"]
            S["XtY"][l, i] = st["XtY"]
        S["n"][i], S["sy"][i], S["syy"][i] = st["n"], st["sy"], st["syy"]
        del hs                                             # 特徴量は保持しない
        print(f"[stats] {i + 1}/{N}", flush=True)

    # ---- fold評価
    rows = []
    for l in range(L):
        rows += evaluate_layer(S, l, folds, args.lambdas_rel)
        print(f"[ridge] layer {l + 1}/{L}", flush=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)


if __name__ == "__main__":
    main()