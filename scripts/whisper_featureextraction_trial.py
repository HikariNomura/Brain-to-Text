# -*- coding: utf-8 -*-
"""
発話音声(試行ごと)をWhisperエンコーダに入力し，層ごとの特徴量をPCAで圧縮して保存する．

入力 : <作業フォルダ>/New_NPY/<被験者><imi>.npy   例: New_NPY/kk_a.npy
       shape = [試行数, ラベル数(8), サンプル数]
       ※EEGと同じ窓(発話開始の前1.0s〜後0.5s = 1.5s)で切り出し済みの音声であること
         ラベルの並びは C++ の condition_map と同じ
         (ame_d, ame_u, nizi_d, nizi_u, momo_d, momo_u, hasi_d, hasi_u)

出力 : <作業フォルダ>/Whisper_Trial/<被験者><imi>/<条件>/layer_<L>.npy
       shape = [試行数, フレーム数, N_PCA] (float64)
       フレーム数 = 窓の長さ[s] * 50  (1.5s なら 75)
       → C++側が「試行iのEEG」と「試行iの特徴量」を対応づけて読み込む

PCAは (条件, 層) ごとに，全試行のフレームをまとめて1回だけ当てはめる．
(教師なしでEEGは使わないので，train/testの分割とは無関係)

必要なライブラリ: numpy, scipy, scikit-learn, torch, transformers
"""

import os
from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from transformers import WhisperFeatureExtractor, WhisperModel

# ---------------------------------------------------------------- 設定
hikensya = ['kk', 'tr', 'ks']
imi = '_a'
word_list = ['ame', 'nizi', 'momo', 'hasi']
oto = ['_d', '_u']
conditions = [w + o for w in word_list for o in oto]  # インデックス0〜7 (C++のcondition_mapと同じ)

AUDIO_SR = 1000       # ★要確認: New_NPY の音声のサンプリング周波数[Hz]
ALLOW_LOW_SR = True   # True にすると AUDIO_SR が低くても(警告のみで)実行する
WHISPER_SR = 16000     # Whisperの入力は16 kHz固定
FRAME_RATE = 50        # Whisperエンコーダの出力は20ms/フレーム = 50 Hz
EXPECTED_WIN_SEC = 1.5 # EEGの切り出し窓 (前0.5s + 後1.0s)

MODEL_NAME = 'openai/whisper-large-v3'
SAVE_LAYERS = list(range(32))  # C++側の num_layers = 32 (layer_0〜layer_31) に合わせる
N_PCA = 10
BATCH_SIZE = 8
# ----------------------------------------------------------------


def preprocess(seg, scale):
    """DC除去済みの1試行分の音声を，Whisper入力用(16kHz, float32, [-1,1])にする"""
    seg = np.clip(seg / scale, -1.0, 1.0)
    if AUDIO_SR != WHISPER_SR:
        g = gcd(AUDIO_SR, WHISPER_SR)
        seg = resample_poly(seg, WHISPER_SR // g, AUDIO_SR // g)
    return seg.astype(np.float32)


@torch.no_grad()
def extract_layers(segs, extractor, encoder, device, dtype, n_frames):
    """
    segs: 16kHz音声のリスト(試行数個)
    戻り値: [試行数, len(SAVE_LAYERS), n_frames, d_model] (float32)
    Whisperは内部で30秒にパディングされ，エンコーダ出力は1500フレームになる．
    先頭の n_frames (=音声の実長ぶん) だけを使う．
    """
    hidden = encoder.config.d_model
    out = np.zeros((len(segs), len(SAVE_LAYERS), n_frames, hidden), dtype=np.float32)
    for s in range(0, len(segs), BATCH_SIZE):
        batch = segs[s:s + BATCH_SIZE]
        feats = extractor(batch, sampling_rate=WHISPER_SR, return_tensors='pt').input_features
        hs = encoder(feats.to(device=device, dtype=dtype), output_hidden_states=True).hidden_states
        for j, layer in enumerate(SAVE_LAYERS):
            out[s:s + len(batch), j] = hs[layer][:, :n_frames, :].float().cpu().numpy()
        print(f'    {min(s + BATCH_SIZE, len(segs))}/{len(segs)} trials')
    return out


def main():
    if AUDIO_SR <= 2000:
        msg = (f'AUDIO_SR={AUDIO_SR} Hz では {AUDIO_SR // 2} Hz 以下の成分しか残っておらず，'
               '母音の情報(フォルマント)がほぼ失われています．')
        if not ALLOW_LOW_SR:
            raise ValueError(msg + ' 試す場合は ALLOW_LOW_SR = True にしてください．')
        print('警告: ' + msg + ' 結果の解釈には注意してください．')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print(f'device={device}, dtype={dtype}, model={MODEL_NAME}')

    extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    encoder = WhisperModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).encoder.to(device).eval()

    for sub in hikensya:
        name = sub + imi
        audio = np.load('./New_NPY/' + name + '_Audio.npy').astype(np.float64)
        print(f'\n=== {name}: audio shape = {audio.shape} ===')
        if audio.ndim != 3 or audio.shape[1] != len(conditions):
            raise ValueError(f'想定は [試行数, {len(conditions)}, サンプル数] ですが {audio.shape} でした')

        n_trials, _, n_samples = audio.shape
        win_sec = n_samples / AUDIO_SR
        n_frames = int(round(win_sec * FRAME_RATE))
        print(f'window = {win_sec:.3f} s -> {n_frames} frames')
        if abs(win_sec - EXPECTED_WIN_SEC) > 1e-3:
            print(f'  警告: 窓の長さがEEG側({EXPECTED_WIN_SEC}s)と違います．切り出し範囲とAUDIO_SRを確認してください')

        # 試行ごとにDC成分を除去し，被験者全体の最大振幅で正規化(試行間の相対的な音量は保つ)
        centered = audio - audio.mean(axis=-1, keepdims=True)
        scale = np.max(np.abs(centered)) + 1e-12

        for ci, cond in enumerate(conditions):
            out_dir = os.path.join('Whisper_Trial', name, cond)
            if all(os.path.exists(os.path.join(out_dir, f'layer_{L}.npy')) for L in SAVE_LAYERS):
                print(f'  [{cond}] skip (already done)')
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f'  [{cond}] extracting...')

            segs = [preprocess(centered[t, ci], scale) for t in range(n_trials)]
            feats = extract_layers(segs, extractor, encoder, device, dtype, n_frames)

            for j, L in enumerate(SAVE_LAYERS):
                f = feats[:, j]                                  # [試行, フレーム, d_model]
                z = PCA(n_components=N_PCA).fit_transform(f.reshape(-1, f.shape[-1]))
                z = z.reshape(n_trials, n_frames, N_PCA).astype(np.float64)
                np.save(os.path.join(out_dir, f'layer_{L}.npy'), z)
            print(f'  [{cond}] saved -> {out_dir}')

            del feats


if __name__ == '__main__':
    main()