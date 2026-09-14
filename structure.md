# ファイル構成

`mTRF_2(0818).py` を C++ に移植したプロジェクトのファイル構成まとめ。

## ディレクトリツリー

```
mtrf_cpp/
├── CMakeLists.txt          # ビルド設定
├── README.md                # 使い方・移植方針の説明
├── structure.md              # 本ファイル
├── .gitignore
├── src/
│   ├── main.cpp              # エントリポイント。全体のパイプラインを記述
│   ├── npy.hpp / npy.cpp     # .npy ファイル読み込み
│   ├── rng.hpp                # numpy互換の乱数生成(試行のシャッフル)
│   ├── pca.hpp                 # PCA(主成分分析)
│   ├── interp.hpp             # 線形補間(50Hz→1000Hz)
│   ├── trf.hpp                 # mTRF(リッジ回帰)の中核ロジック
│   ├── stats.hpp               # 相関係数・決定係数(R²)
│   ├── butterworth.hpp        # バターワースローパスフィルタ(オプション機能)
│   └── csv.hpp                 # 結果CSVの書き出し
└── third_party/
    └── eigen/                 # 線形代数ライブラリ Eigen(ヘッダオンリー、同梱)
```

## 各ファイルの役割

### `CMakeLists.txt`
C++17でビルドする設定。`third_party/eigen` をインクルードパスに追加するだけで、
Eigen自体は追加のビルドやリンクは不要(ヘッダオンリーライブラリ)。

### `src/main.cpp`
Pythonスクリプト全体の処理順序をそのまま踏襲したエントリポイント。

- 設定値(`hikensya`, `imi`, `word_list`, `regularization`, `tmin`/`tmax` など)
  はPython版の定数をそのままC++の`const`として定義
- word(2種)→ layer(2種)→ channel(21ch)→ fold(20分割)の4重ループ構造
- `.npy`のパスや`EEG_lowpass`のON/OFFは環境変数で上書き可能(後述)
- 各wordごとに、全layer・全channel・全foldの結果をまとめて一度だけCSVに保存

### `src/npy.hpp` / `src/npy.cpp`
NumPyの`.npy`形式のパーサ。マジックバイト・ヘッダ(dtype, shape,
fortran_order)を読み取り、任意次元の配列を`double`のフラット配列として読み込む。
`float32/float64/int8~64/uint8~64/bool` に対応。

### `src/rng.hpp`
`np.random.default_rng(0)` (PCG64ビット生成器) と、その `.permutation()`
(Fisher–Yatesシャッフル) を独自に再実装したもの。**seed=0固定**
(元スクリプトが常にseed 0しか使わないため)。numpy 2.5.1の実際の出力と
ビット単位で一致することを検証済み。これにより、train/tune/test
の試行の割り振りがPython版と完全に一致する。

### `src/pca.hpp`
`sklearn.decomposition.PCA(n_components=k).fit_transform(X)` 相当。
Eigenの`BDCSVD`で決定論的なフルSVDを行い、sklearnの符号規約
(`svd_flip`)を再現。`sklearn 1.9.0`の`svd_solver="full"`と誤差1e-15で一致検証済み。

※元のPythonコードは`random_state`未指定のため、実際には非決定論的な
randomized SVDが使われており、Python自体も実行のたびに結果が変わる
(詳細はREADME参照)。

### `src/interp.hpp`
`scipy.interpolate.interp1d(..., fill_value="extrapolate")` 相当の
線形補間・外挿。Whisper特徴量(50Hz)をEEGのサンプリングレート(1000Hz)
に合わせるために使用。scipyと誤差1e-15で一致検証済み。

### `src/trf.hpp`
mTRF(multivariate Temporal Response Function)のリッジ回帰の中核。

- `build_lag_matrix()`: 時間ラグ付き計画行列の構築(`mtrf`パッケージの
  `lag_matrix()`相当)
- `build_trf_context()` / `TrfContext`: 共分散行列の計算とリッジ回帰の
  事前分解(LDLT分解)。**全チャンネル・全foldで使い回せる**ように
  設計されており、これが実行時間を現実的な範囲に収めている一番の工夫点
  (詳細はコード内コメントとREADME参照)
- 実際に`mtrf`パッケージ(PyPI)をインストールして`TRF.train()`/`.predict()`
  と比較し、重み・予測値が一致することを検証済み

### `src/stats.hpp`
`scipy.stats.pearsonr` 相当のピアソン相関係数と、
`sklearn.metrics.r2_score` 相当の決定係数(R²)。

### `src/butterworth.hpp`
`scipy.signal.butter(N=4, Wn, btype="low")` + `scipy.signal.filtfilt()`
相当の4次バターワースローパスフィルタ(ゼロ位相)。

- バイリニア変換によるフィルタ設計
- Direct Form II Transposed による `lfilter`
- `lfilter_zi` による初期条件つきの`filtfilt`(奇関数パディング)
- scipyと数値照合済み(係数誤差 ~1.8e-15、filtfilt結果誤差 ~9.3e-13)

元スクリプトの`EEG_lowpass`フラグに相当する機能。`MTRF_EEG_LOWPASS`
環境変数でビルドし直さずにON/OFFを切り替えられる(後述)。

### `src/csv.hpp`
結果(`layer, channel, fold, train_n, tune_n, test_n, mean_r, mean_r2`)
をCSVに書き出す。元コードにあった「チャンネルごとに同じファイルを
上書きしてしまうバグ」を修正し、全結果を1回でまとめて書き出す仕様に
変更済み。

### `third_party/eigen/`
線形代数ライブラリ [Eigen](https://eigen.tuxfamily.org/) 3.4.0
(ヘッダオンリー)をそのまま同梱。PCA・共分散行列・リッジ回帰の
連立方程式の求解などに使用。

## データフロー(概要)

```
.npy (EEG)  ──┐
              ├─→ [PCA] whisper特徴量を10次元に圧縮
.npy (Whisper)┘        ↓
                  [線形補間] 50Hz → 1000Hz
                        ↓
                  [ラグ行列構築] tmin~tmax の時間ラグ特徴量
                        ↓
                  [共分散行列 + リッジ回帰] チャンネル・foldで使い回し
                        ↓
                  [予測 → 相関係数/R²] test試行ごとに評価
                        ↓
                  [CSV書き出し] word単位で1回
```

## 実行時に切り替えられる環境変数

| 環境変数 | 説明 | デフォルト |
|---|---|---|
| `MTRF_EEG_ROOT` | EEGの`.npy`が置かれているディレクトリ | 元コードと同じハードコードパス |
| `MTRF_WHISPER_ROOT` | Whisper特徴量の`.npy`が置かれているルート | 元コードと同じハードコードパス |
| `MTRF_EEG_LOWPASS` | バターワースローパスのON/OFF (`1`/`0`, `true`/`false`, `on`/`off`, `yes`/`no`) | `false`(元コードと同じ) |
