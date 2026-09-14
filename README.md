# mtrf_cpp

`mTRF_2(0818).py` の C++17 移植版(Whisper特徴量を刺激、EEG応答を目的変数と
した順方向mTRFリッジ回帰解析。train/tune/test を20分割でローテーションする)。

ファイル構成の詳細は [structure.md](structure.md) を参照。

## ビルド方法

CMake と C++17 対応コンパイラが必要(MinGW-w64 g++ 15.2で動作確認済み)。
Eigen は `third_party/eigen` に同梱済み(ヘッダオンリーなので追加のビルド
手順は不要)。

```bash
mkdir build && cd build
cmake -G "MinGW Makefiles" ..
mingw32-make -j4
./mtrf.exe
```

(`src/rng.hpp` で `__int128` を使用しているため、GCC/Clangであれば
どのCMakeジェネレータでも問題ない(例: MSVC/Ninjaで
`cmake .. && cmake --build . --config Release`)。ただしMSVCの
`cl.exe`は`__int128`に対応していないため、Windowsでは
clang-clかMinGWでビルドすること。)

デフォルトでは、Pythonスクリプトと同じハードコードされたパスから
読み込む:
- `C:/Users/Kurisu/Desktop/Working2024/NPY/03_PreProCutEEG/<hikensya><imi>.npy`
- `C:/Users/Kurisu/Desktop/R8 hikari/Whisper/Feature_extraction/<word>/layer_<n>.npy`

テストや別環境での実行のために、これらのルートは環境変数
`MTRF_EEG_ROOT` / `MTRF_WHISPER_ROOT` で上書き可能。

出力CSVは `<today>/mtrf_90_5_5_<word>.csv` に書き出される(Python版と同じ。
`today`は現状`main.cpp`内で`"0818"`にハードコードされており、これも
スクリプトに合わせてある)。

### バターワースローパスフィルタのON/OFF

元スクリプトの`EEG_lowpass`フラグに相当する機能。デフォルトはPython版と
同じ`false`(無効)。ビルドし直さずに、環境変数`MTRF_EEG_LOWPASS`で
切り替え可能:

```bash
MTRF_EEG_LOWPASS=1 ./mtrf.exe   # 有効化(4次20Hzバターワース + filtfilt)
MTRF_EEG_LOWPASS=0 ./mtrf.exe   # 無効化(デフォルトと同じ)
```

`1`/`0`, `true`/`false`, `on`/`off`, `yes`/`no`(大文字小文字区別なし)を
受け付ける。未設定時はコード内の`EEG_lowpass_default`(`false`)に従う。

## 「忠実な移植で検証済みの部分」と「意図的にドキュメント化した近似」

数値計算の中核部分は、実際のPythonライブラリ(scipy 1.18、
scikit-learn 1.9.0、`mtrf`パッケージ)をローカルにインストールし、
小さな参照ケースで出力を突き合わせて個別に検証済み:

- **リッジ回帰によるTRFモデル**(`src/trf.hpp`): ラグ行列の構築・共分散の
  蓄積・リッジ回帰の求解を、実際にスクリプトがimportしている
  `mtrf.model.TRF(direction=1).train(...)` / `.predict(...)` と直接比較。
  ランダムなテストケースで予測値・バイアス・重みが ~1e-10
  (倍精度)まで一致することを確認済み。
- **PCA**(`src/pca.hpp`): `sklearn.decomposition.PCA(svd_solver="full")`
  と ~1e-15 で一致。ただし`svd_solver="auto"`に関する注意点は下記参照。
- **補間**(`src/interp.hpp`): `scipy.interpolate.interp1d(...,
  fill_value="extrapolate")` と ~1e-15 で一致。
- **ピアソン相関係数 / R²**(`src/stats.hpp`): `scipy.stats.pearsonr` /
  `sklearn.metrics.r2_score` と倍精度で完全一致。
- **試行のシャッフル**(`src/rng.hpp`): numpyのPCG64ビット生成器と
  `Generator.permutation()`のFisher–Yatesシャッフルをゼロから再実装し、
  スクリプトが常に使う`seed=0`固定でハードコード。numpy 2.5.1の
  `np.random.default_rng(0).permutation(np.arange(100))`の実際の出力と
  **ビット単位で完全一致**することを検証済み。つまりtrain/tune/testの
  試行の割り振りはPython版と1試行単位で完全に一致する。
- **バターワースローパスフィルタ**(`src/butterworth.hpp`):
  `scipy.signal.butter(N=4, Wn, btype="low")` +
  `scipy.signal.filtfilt(...)` と比較し、フィルタ係数の誤差 ~1.8e-15、
  filtfilt出力の誤差 ~9.3e-13 で一致することを検証済み。

以下の2点は、意図的に「1行1行の逐語訳」にはしていない。逐語訳自体が
成立しない、あるいは計算量的に非現実的なため。理由はコード中の該当箇所の
コメントにも記載している:

1. **PCAのソルバー。** スクリプトは`random_state`を指定せずに
   `PCA(n_components=10).fit_transform(...)`を呼んでいる。この入力形状
   では、scikit-learnの`svd_solver="auto"`は*randomized SVD*を選択し、
   これはnumpyのグローバルな(シード未設定の)乱数状態から乱数を
   引く。つまりPythonスクリプト自体が、実行するたびに異なるPCA結果を
   返す。「正しい1回の実行結果」というものがそもそも存在しないため、
   ビット単位で一致させることは目標として成立しない。そのためこの
   移植版では常に決定論的なフルSVDを使用している(上記で検証済み。
   randomized SVDが近似しようとしている対象そのもの)。後段の数値は
   非常に近い値にはなるが、完全には一致しない。

2. **試行ごとの共分散の蓄積。** スクリプトでは、あるword/layerの
   全ての試行が*同じ*刺激行列`X`に対して学習される(試行ごとに違うのは
   EEG応答`Y`のみ)。`mtrf`パッケージの`covariance_matrices()`は、
   それでも学習用の約90試行すべてをループし、毎回ラグ行列の積を
   計算・蓄積する。これは代数的には完全に冗長であり(しかも
   愚直に実行すると1日程度かかる計算量になる: 約1680通りの
   channel/fold組み合わせ × 最大90試行 × 2011×2011行列の積、を
   すべて計算することになる)。`X`が試行に依らず一定であるため、
   この蓄積は実数演算上*厳密に*
   `x_lag.T @ x_lag`(試行数は打ち消し合う)と
   `x_lag.T @ (学習用試行群でのYの平均)` に帰着する。この移植版では
   ラグ行列と`(cov_xx + regmat)`の分解をword/layerごとに1回だけ計算し、
   全チャンネル・全foldで使い回している。これが移植版を現実的な
   時間で実行可能にした最大のポイント。導出は`src/trf.hpp`のコメント
   を参照。

## CSV上書きバグ: 修正済み

元スクリプトでは、`results_df`を作成してword単位のCSV
(`today/mtrf_90_5_5_<word>.csv`)を書き出す処理が`channel_idx`ループの
*内側*にあり、しかもファイル名にチャンネル(やlayer)の情報が含まれて
いなかった。そのため書き込むたびに直前の内容を上書きしてしまい、
スクリプト終了時にはチャンネル20の20行分の結果しかディスク上に
残らなかった。

この移植版では修正済み: `FoldResult`に`layer`と`channel`の列を追加し、
あるwordについて計算されたすべての(layer, channel, fold)の行を
メモリ上に集約したうえで、そのwordの全layer・全channelの処理が
終わった後に`today/mtrf_90_5_5_<word>.csv`を1回だけ書き出すように
変更した。結果として、20行ではなく
`layer数 × 21チャンネル × 20fold`行(デフォルトの2layer構成では840行)
がすべて保存される。

## パフォーマンス

上記の共分散使い回しの工夫のおかげで、フル実行(2 words × 2 layers ×
21 channels × 20 folds、応答1000サンプル、刺激特徴量10次元、
201ラグ = 2011×2011のリッジ回帰系)は数時間・数日ではなく数秒程度で
完了する。
