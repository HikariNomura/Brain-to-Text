// C++ port of mTRF pipeline: 発話版 (試行ごとに刺激が異なる 100対100 版)
//
// 聴取版との違い:
//   - 聴取版: 刺激(音源)は1つ → 100試行の脳波を「平均」してから1本のTRFを学習・評価
//   - 発話版: 試行ごとに発話(=Whisper特徴量)が違う → 試行を平均できない
//       ・train: 全trainトライアルの(刺激, 脳波)ペアをまとめて1つのTRFを学習 (正規方程式を試行ごとに足し合わせる)
//       ・tune : tune試行を1試行ずつ予測 → 試行ごとのrの平均が最大の正則化を選ぶ (チャンネルごと, foldごと)
//       ・test : test試行を1試行ずつ予測 → 試行ごとのr, r2を求めて平均 (= そのfoldのr, r2)
//   - 5fold分のr, r2の「平均」と「標準偏差」を記録する (CSV形式は聴取版と同じ)
//
// 入力:
//   EEG     : <eeg_root>/<被験者><imi>.npy        shape [試行, 8条件, 21ch, 1500]  (発話開始の前1s〜後0.5s)
//   Whisper : <whisper_root>/<被験者><imi>/<条件>/layer_<L>.npy   shape [試行, 75, 10]
//             (Whisper_FeatureExtraction_Trial.py の出力. PCA済み)
//
// 設計方針:
//   - 条件(ame_d, ame_u, ...)ごとに独立して5-fold分割
//   - 切片は学習データの平均で中心化して扱う (正則化されない切片)
//   - Python版とのbit-exact再現性は目的としない
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "npy.hpp"
#include "rng.hpp"
#include "interp.hpp"
#include "stats.hpp"
#include "butterworth.hpp"

namespace fs = std::filesystem;

namespace {

// ---- configuration ----
// const std::vector<std::string> hikensya = {"kk", "tr", "ks"};
const std::vector<std::string> hikensya = {"mm"};
const std::vector<std::string> imi = {"_a", "_n"};
const int imi_choice = 0;

const std::vector<std::string> word_list = {"ame", "nizi", "momo", "hasi"};
const std::vector<std::string> oto = {"_d", "_u"};

const bool EEG_lowpass = false;
const bool EEG_bandpass = false;

// 聴取版の結果(例: "1014")を上書きしないよう別名にしている
const std::string out_tag = "1120";

const int eeg_sr = 1000;
const int whisper_sr = 50;
const int num_layers = 32;   // layer_0 〜 layer_31

// 刺激→脳波の遅れ(秒). 正の値 = 刺激が脳波より先 (聴取・自分の声のフィードバック向け)
// 発話に先行する運動関連の活動を見たいときは負のラグ(例: tmin=-0.2, tmax=0.0)にする
const double tmin = -0.5;
const double tmax = 1.5;
// ラグの刻み[サンプル]. 特徴量は50 Hz(20ms)なので5msでも情報はほぼ落ちない.
// 1にすると従来どおり1msごと(列数が5倍 → Gram行列の計算量は約25倍, 連立方程式は約125倍)
const int lag_step = 20;

// ---- fold構成 & 正則化候補 ----
// 聴取版は「80試行の平均」を1回学習していたが，発話版は80試行ぶんの行数を使うので
// Gram行列の規模が大きくなる. 最適な正則化が上側にずれるため候補を 1e6, 1e7 まで拡張している.
// mean_reg が候補の端(1e7)に張り付く場合はさらに拡張すること.
const int TEST_N = 8;
const int TUNE_N = 2;
const std::vector<double> reg_candidates = {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};

struct FoldSizing {
    int chunk_size;
    int test_n;
    int tune_n;
    int n_folds;
};

// n_trialsの約数の中から，target_chunk_sizeに一番近いものを選ぶ.
// (n_trialsが100と異なる被験者がいても自動で対応するため)
FoldSizing choose_fold_sizing(int n_trials, int target_test_n, int target_tune_n) {
    int target_chunk_size = target_test_n + target_tune_n;

    int best_chunk = -1;
    int best_diff = std::numeric_limits<int>::max();
    for (int c = 1; c <= n_trials; c++) {
        if (n_trials % c != 0) continue;
        int diff = std::abs(c - target_chunk_size);
        if (diff < best_diff) {
            best_diff = diff;
            best_chunk = c;
        }
    }
    if (best_chunk < 2) {
        throw std::runtime_error("choose_fold_sizing: could not find a usable chunk size for n_trials=" +
                                  std::to_string(n_trials));
    }

    double tune_ratio = (double)target_tune_n / (double)target_chunk_size;
    int tune_n = std::max(1, (int)std::round(best_chunk * tune_ratio));
    int test_n = best_chunk - tune_n;

    int n_folds = n_trials / best_chunk;
    return FoldSizing{best_chunk, test_n, tune_n, n_folds};
}

std::map<std::string, int> make_condition_map() {
    return {
        {"ame_d", 0}, {"ame_u", 1}, {"nizi_d", 2}, {"nizi_u", 3},
        {"momo_d", 4}, {"momo_u", 5}, {"hasi_d", 6}, {"hasi_u", 7},
    };
}

// 指定条件の全チャネルを一括抽出する.
// 戻り値: (n_trials, n_channels * n_time) 行列.
// 列レイアウト: [ch0_t0..ch0_tT-1, ch1_t0..ch1_tT-1, ..., ch_{C-1}_t0..]
Eigen::MatrixXd extract_condition_all_channels(const NpyArray& eeg, int condition_idx, int n_time) {
    if (eeg.ndim() != 4) throw std::runtime_error("expected eeg to be a 4-D array");
    size_t n_trials = eeg.shape[0];
    size_t n_cond   = eeg.shape[1];
    size_t n_chan   = eeg.shape[2];
    size_t T        = eeg.shape[3];
    if ((size_t)condition_idx >= n_cond || (size_t)n_time > T)
        throw std::runtime_error("extract_condition_all_channels: index out of range");
    auto strides = eeg.strides();
    Eigen::MatrixXd Y_all((Eigen::Index)n_trials, (Eigen::Index)(n_chan * n_time));
    for (size_t trial = 0; trial < n_trials; trial++) {
        size_t base_cond = trial * strides[0] + (size_t)condition_idx * strides[1];
        for (size_t ch = 0; ch < n_chan; ch++) {
            size_t base = base_cond + ch * strides[2];
            for (int t = 0; t < n_time; t++)
                Y_all((Eigen::Index)trial, (Eigen::Index)(ch * n_time + t)) = eeg.data[base + t];
        }
    }
    return Y_all;
}

NpyArray apply_eeg_lowpass(const NpyArray& eeg, int sr) {
    auto [b, a] = butterworth::design_lowpass(4, 20.0 / (sr / 2.0));
    NpyArray out = eeg;
    size_t T = eeg.shape[3];
    size_t n_series = eeg.data.size() / T;
    std::vector<double> buf(T);
    for (size_t s = 0; s < n_series; s++) {
        size_t base = s * T;
        for (size_t t = 0; t < T; t++) buf[t] = eeg.data[base + t];
        std::vector<double> filtered = butterworth::filtfilt(b, a, buf);
        for (size_t t = 0; t < T; t++) out.data[base + t] = filtered[t];
    }
    return out;
}

NpyArray apply_eeg_bandpass(const NpyArray& eeg, int sr) {
    double nyquist = sr / 2.0;
    auto [b, a] = butterworth::design_bandpass(3, 0.5 / nyquist, 8.0 / nyquist);
    NpyArray out = eeg;
    size_t T = eeg.shape[3];
    size_t n_series = eeg.data.size() / T;
    std::vector<double> buf(T);
    for (size_t s = 0; s < n_series; s++) {
        size_t base = s * T;
        for (size_t t = 0; t < T; t++) buf[t] = eeg.data[base + t];
        std::vector<double> filtered = butterworth::filtfilt(b, a, buf);
        for (size_t t = 0; t < T; t++) out.data[base + t] = filtered[t];
    }
    return out;
}

struct FoldSplit {
    std::vector<int> train_trials;
    std::vector<int> tune_trials;
    std::vector<int> test_trials;
};

std::vector<FoldSplit> make_fold_splits(int n_trials, int test_n, int tune_n) {
    int chunk_size = test_n + tune_n;
    if (n_trials % chunk_size != 0) {
        throw std::runtime_error("make_fold_splits: n_trials must be divisible by test_n+tune_n");
    }
    int n_folds = n_trials / chunk_size;

    std::vector<int> perm = mtrf::permutation_seed0(n_trials);

    std::vector<FoldSplit> splits(n_folds);
    for (int fold = 0; fold < n_folds; fold++) {
        int chunk_start = fold * chunk_size;

        std::vector<int> test_trials(perm.begin() + chunk_start, perm.begin() + chunk_start + test_n);
        std::vector<int> tune_trials(perm.begin() + chunk_start + test_n,
                                      perm.begin() + chunk_start + chunk_size);

        std::vector<int> train_trials;
        train_trials.reserve(n_trials - chunk_size);
        for (int i = 0; i < n_trials; i++) {
            if (i < chunk_start || i >= chunk_start + chunk_size) train_trials.push_back(perm[i]);
        }

        splits[fold].test_trials = std::move(test_trials);
        splits[fold].tune_trials = std::move(tune_trials);
        splits[fold].train_trials = std::move(train_trials);
    }
    return splits;
}

double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

double std_of(const std::vector<double>& v, double mean) {
    double acc = 0.0;
    for (double x : v) acc += (x - mean) * (x - mean);
    return std::sqrt(acc / (double)v.size());
}

// ---------------------------------------------------------------------------
// 発話版で追加した部分
// ---------------------------------------------------------------------------

// Whisper特徴量(PCA済み) [試行, フレーム, 次元] を試行ごとの行列 (フレーム x 次元) に分けて読む
std::vector<Eigen::MatrixXd> load_whisper_trials(const std::string& path, int n_trials, int n_frames) {
    NpyArray a = load_npy(path);
    if (a.ndim() != 3)
        throw std::runtime_error("whisper feature must be 3-D [trial, frame, pca]: " + path);
    if ((int)a.shape[0] != n_trials)
        throw std::runtime_error("trial count mismatch (EEG=" + std::to_string(n_trials) +
                                  ", whisper=" + std::to_string(a.shape[0]) + "): " + path);
    if ((int)a.shape[1] != n_frames)
        throw std::runtime_error("frame count mismatch (expected " + std::to_string(n_frames) +
                                  ", got " + std::to_string(a.shape[1]) + "): " + path);
    const size_t F = a.shape[1], D = a.shape[2];
    std::vector<Eigen::MatrixXd> out((size_t)n_trials);
    for (int i = 0; i < n_trials; i++) {
        out[i].resize((Eigen::Index)F, (Eigen::Index)D);
        for (size_t f = 0; f < F; f++)
            for (size_t d = 0; d < D; d++)
                out[i]((Eigen::Index)f, (Eigen::Index)d) = a.data[((size_t)i * F + f) * D + d];
    }
    return out;
}

// ラグ付きデザイン行列: L[t, li*F + f] = X[t - lags[li], f]  (範囲外は0)
// lag > 0: 刺激が脳波より lag サンプル先行. lag < 0 も使用可.
Eigen::MatrixXd build_lagged(const Eigen::MatrixXd& X, const std::vector<int>& lags) {
    const Eigen::Index T = X.rows(), F = X.cols();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(T, (Eigen::Index)lags.size() * F);
    for (size_t li = 0; li < lags.size(); li++) {
        const Eigen::Index lag = lags[li];
        const Eigen::Index t_begin = std::max<Eigen::Index>(0, lag);
        const Eigen::Index t_end = std::min<Eigen::Index>(T, T + lag); // 排他的
        if (t_end <= t_begin) continue;
        L.block(t_begin, (Eigen::Index)li * F, t_end - t_begin, F) =
            X.block(t_begin - lag, 0, t_end - t_begin, F);
    }
    return L;
}

// 試行ごとの十分統計量 (layerごとに1回だけ計算し，foldでは足し合わせるだけにする)
struct TrialStats {
    Eigen::MatrixXd G;    // L^T L          (P x P)
    Eigen::VectorXd sx;   // L^T 1          (P)
    Eigen::MatrixXd xy;   // L^T Y          (P x C)
    Eigen::VectorXd sy;   // Y^T 1          (C)
};

struct AggResult {
    int layer;
    int channel;
    int n_folds;
    int train_n;
    int tune_n;
    int test_n;
    double mean_r;
    double std_r;
    double mean_r2;
    double std_r2;
    double mean_reg;
};

void write_agg_csv(const std::string& path, const std::vector<AggResult>& rows) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << "layer,channel,n_folds,train_n,tune_n,test_n,mean_r,std_r,mean_r2,std_r2,mean_reg\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        f << r.layer << "," << r.channel << "," << r.n_folds << "," << r.train_n << ","
          << r.tune_n << "," << r.test_n << "," << r.mean_r << "," << r.std_r << ","
          << r.mean_r2 << "," << r.std_r2 << "," << r.mean_reg << "\n";
    }
}

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cout << std::unitbuf;

    // ラグ(サンプル)は全ループ外で1回だけ構築
    std::vector<int> lags;
    for (int l = (int)std::lround(tmin * eeg_sr); l <= (int)std::lround(tmax * eeg_sr); l += lag_step)
        lags.push_back(l);
    std::cout << "lags: " << lags.size() << " (" << lags.front() << " .. " << lags.back()
              << " samples, step " << lag_step << ")\n";

    auto condition_map = make_condition_map();

    const char* eeg_root_env     = std::getenv("MTRF_EEG_ROOT");
    const char* whisper_root_env = std::getenv("MTRF_WHISPER_ROOT");
    const std::string eeg_root =
        eeg_root_env ? eeg_root_env : "C:/Users/Kurisu/Desktop/R8-hikari/Brain-to-Text/New_NPY/rokuon";
        
    // Whisper_FeatureExtraction_Trial.py の出力先 (<作業フォルダ>/Whisper_Trial) に合わせること
    const std::string whisper_root =
        whisper_root_env ? whisper_root_env : "C:/Users/Kurisu/Desktop/R8-hikari/Brain-to-Text/Whisper_Trial/1120";
        

    for (size_t k = 0; k < hikensya.size(); k++) {
        const std::string subject = hikensya[k] + imi[imi_choice];
        const std::string eeg_path = eeg_root + "/" + subject + ".npy";

        std::cout << "Loading EEG: " << eeg_path << "\n";
        NpyArray eeg = load_npy(eeg_path);
        if (eeg.ndim() != 4) throw std::runtime_error("EEG must be 4-D [trial, cond, ch, time]: " + eeg_path);

        NpyArray eeg_filtered;
        if (EEG_bandpass) {
            eeg_filtered = apply_eeg_bandpass(eeg, eeg_sr);
        } else if (EEG_lowpass) {
            eeg_filtered = apply_eeg_lowpass(eeg, eeg_sr);
        }
        const NpyArray& eeg_used = (EEG_bandpass || EEG_lowpass) ? eeg_filtered : eeg;

        const int n_trials = (int)eeg_used.shape[0];
        const int C = (int)eeg_used.shape[2];                 // チャンネル数 (21)
        const int n_time = (int)eeg_used.shape[3];            // 1500
        const int n_frames = (int)std::lround((double)n_time * whisper_sr / eeg_sr); // 75
        std::cout << "EEG shape: (" << eeg_used.shape[0] << ", " << eeg_used.shape[1] << ", "
                  << eeg_used.shape[2] << ", " << eeg_used.shape[3] << ")  -> frames=" << n_frames << "\n";

        // 時間軸は定数 → 被験者ごとに1回
        std::vector<double> eeg_time(n_time);
        for (int t = 0; t < n_time; t++) eeg_time[t] = (double)t / eeg_sr;
        std::vector<double> whisper_time(n_frames);
        for (int t = 0; t < n_frames; t++) whisper_time[t] = (double)t / whisper_sr;

        FoldSizing sizing = choose_fold_sizing(n_trials, TEST_N, TUNE_N);
        std::cout << "Fold sizing: chunk=" << sizing.chunk_size
                  << " (test=" << sizing.test_n << ", tune=" << sizing.tune_n << ")"
                  << ", n_folds=" << sizing.n_folds << "\n";
        std::vector<FoldSplit> splits = make_fold_splits(n_trials, sizing.test_n, sizing.tune_n);
        const int train_n = n_trials - sizing.chunk_size;
        const size_t nreg = reg_candidates.size();

        for (const std::string& w : word_list) {
        for (const std::string& o : oto) {
            const std::string word = w + o;
            std::cout << std::string(60, '=') << "\n" << subject << " / " << word << "\n"
                      << std::string(60, '=') << "\n";
            const int condition_idx = condition_map.at(word);

            // 試行ごとの脳波 (n_time x C). layerによらず不変なので1回だけ構築
            Eigen::MatrixXd Y_all = extract_condition_all_channels(eeg_used, condition_idx, n_time);
            std::vector<Eigen::MatrixXd> Y_trials((size_t)n_trials, Eigen::MatrixXd(n_time, C));
            for (int i = 0; i < n_trials; i++)
                for (int c = 0; c < C; c++)
                    for (int t = 0; t < n_time; t++)
                        Y_trials[i](t, c) = Y_all(i, (Eigen::Index)c * n_time + t);
            Y_all.resize(0, 0);

            std::vector<AggResult> word_results;

            // 試行×チャンネルごとのtest相関 (testに使われなかった試行はNaN)
            NpyArray trial_r;
            trial_r.shape = {(size_t)num_layers, (size_t)C, (size_t)n_trials};
            trial_r.data.assign((size_t)num_layers * C * n_trials, std::numeric_limits<double>::quiet_NaN());

            for (int layer = 0; layer < num_layers; layer++) {
                const std::string whisper_path =
                    whisper_root + "/" + subject + "/" + word + "/layer_" + std::to_string(layer) + ".npy";
                std::vector<Eigen::MatrixXd> W = load_whisper_trials(whisper_path, n_trials, n_frames);
                const int D = (int)W[0].cols();
                const Eigen::Index P = (Eigen::Index)lags.size() * D;

                // 試行ごとの刺激を脳波の時間軸に補間 → 十分統計量を計算
                std::vector<Eigen::MatrixXd> X_trials((size_t)n_trials);
                std::vector<TrialStats> ts((size_t)n_trials);
                for (int i = 0; i < n_trials; i++) {
                    X_trials[i] = linear_interp_extrapolate(whisper_time, W[i], eeg_time); // (n_time x D)
                    Eigen::MatrixXd L = build_lagged(X_trials[i], lags);                   // (n_time x P)

                    ts[i].G = Eigen::MatrixXd::Zero(P, P);
                    ts[i].G.selfadjointView<Eigen::Lower>().rankUpdate(L.transpose());
                    ts[i].G.triangularView<Eigen::StrictlyUpper>() = ts[i].G.transpose().eval();
                    ts[i].sx = L.colwise().sum().transpose();
                    ts[i].xy = L.transpose() * Y_trials[i];
                    ts[i].sy = Y_trials[i].colwise().sum().transpose();
                }

                std::vector<std::vector<double>> fold_r((size_t)C), fold_r2((size_t)C), fold_reg((size_t)C);

                for (size_t fold = 0; fold < splits.size(); fold++) {
                    const FoldSplit& sp = splits[fold];

                    // ---- train: 全trainトライアルの統計量を合算し，中心化 ----
                    Eigen::MatrixXd Sxx = Eigen::MatrixXd::Zero(P, P);
                    Eigen::VectorXd Sx  = Eigen::VectorXd::Zero(P);
                    Eigen::MatrixXd Sxy = Eigen::MatrixXd::Zero(P, C);
                    Eigen::VectorXd Sy  = Eigen::VectorXd::Zero(C);
                    double N = 0.0;
                    for (int t : sp.train_trials) {
                        Sxx += ts[t].G;
                        Sx  += ts[t].sx;
                        Sxy += ts[t].xy;
                        Sy  += ts[t].sy;
                        N   += (double)n_time;
                    }
                    Eigen::VectorXd x_mean = Sx / N;
                    Eigen::VectorXd y_mean = Sy / N;
                    Eigen::MatrixXd Cxx = Sxx - N * x_mean * x_mean.transpose();
                    Eigen::MatrixXd Cxy = Sxy - N * x_mean * y_mean.transpose();

                    // 正則化候補ごとに全チャンネルぶんの重みを一括で解く
                    std::vector<Eigen::MatrixXd> Wr(nreg);
                    std::vector<Eigen::RowVectorXd> br(nreg);
                    for (size_t ci = 0; ci < nreg; ci++) {
                        Eigen::MatrixXd A = Cxx;
                        A.diagonal().array() += reg_candidates[ci];
                        Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
                        Wr[ci] = ldlt.solve(Cxy);                                   // (P x C)
                        br[ci] = (y_mean - Wr[ci].transpose() * x_mean).transpose(); // 切片 (1 x C)
                    }

                    // ---- tune: 試行ごとのrの平均が最大の正則化をチャンネルごとに選ぶ ----
                    std::vector<Eigen::VectorXd> tune_r(nreg, Eigen::VectorXd::Zero(C));
                    for (int t : sp.tune_trials) {
                        Eigen::MatrixXd L = build_lagged(X_trials[t], lags);
                        for (size_t ci = 0; ci < nreg; ci++) {
                            Eigen::MatrixXd Yh = L * Wr[ci];
                            Yh.rowwise() += br[ci];
                            for (int c = 0; c < C; c++) {
                                Eigen::VectorXd yt = Y_trials[t].col(c);
                                Eigen::VectorXd yp = Yh.col(c);
                                tune_r[ci](c) += pearson_r(yt, yp) / (double)sp.tune_trials.size();
                            }
                        }
                    }
                    std::vector<int> best((size_t)C, 0);
                    for (int c = 0; c < C; c++) {
                        double bv = -std::numeric_limits<double>::infinity();
                        for (size_t ci = 0; ci < nreg; ci++) {
                            if (tune_r[ci](c) > bv) { bv = tune_r[ci](c); best[c] = (int)ci; }
                        }
                    }

                    // ---- test: 1試行ずつ予測して r, r2 を求め，試行平均 ----
                    std::vector<double> r_sum((size_t)C, 0.0), r2_sum((size_t)C, 0.0);
                    for (int t : sp.test_trials) {
                        Eigen::MatrixXd L = build_lagged(X_trials[t], lags);
                        std::vector<Eigen::MatrixXd> Yh(nreg);
                        std::vector<char> have(nreg, 0);
                        for (int c = 0; c < C; c++) {
                            const int ci = best[c];
                            if (!have[ci]) {
                                Yh[ci] = L * Wr[ci];
                                Yh[ci].rowwise() += br[ci];
                                have[ci] = 1;
                            }
                            Eigen::VectorXd yt = Y_trials[t].col(c);
                            Eigen::VectorXd yp = Yh[ci].col(c);
                            double r  = pearson_r(yt, yp);
                            double r2 = r2_score(yt, yp);
                            r_sum[c]  += r;
                            r2_sum[c] += r2;
                            trial_r.data[((size_t)layer * C + c) * n_trials + t] = r;
                        }
                    }
                    const double n_test = (double)sp.test_trials.size();
                    for (int c = 0; c < C; c++) {
                        fold_r[c].push_back(r_sum[c] / n_test);
                        fold_r2[c].push_back(r2_sum[c] / n_test);
                        fold_reg[c].push_back(reg_candidates[best[c]]);
                    }
                }

                // ---- fold間の平均・標準偏差を記録 ----
                double layer_mean_r = 0.0;
                for (int c = 0; c < C; c++) {
                    double mr   = mean_of(fold_r[c]);
                    double sr   = std_of(fold_r[c], mr);
                    double mr2  = mean_of(fold_r2[c]);
                    double sr2  = std_of(fold_r2[c], mr2);
                    double mreg = mean_of(fold_reg[c]);
                    layer_mean_r += mr / C;
                    word_results.push_back(AggResult{
                        layer, c, (int)splits.size(), train_n, sizing.tune_n, sizing.test_n,
                        mr, sr, mr2, sr2, mreg});
                }
                std::cout << "Layer " << std::setw(2) << layer << ": channel-mean r = "
                          << std::fixed << std::setprecision(4) << layer_mean_r << "\n";
            }

            const std::string out_dir = "../data_speech/" + out_tag + "/" + hikensya[k] + "/";
            const std::string out_path = out_dir + "mtrf_agg_" + word + ".csv";
            write_agg_csv(out_path, word_results);
            std::cout << "Saved: " << out_path << " (" << word_results.size() << " rows)\n";

            const std::string trial_out_path = out_dir + "mtrf_trial_r_" + word + ".npy";
            save_npy(trial_out_path, trial_r);
            std::cout << "Saved: " << trial_out_path << "\n";
        }
        }
    }

    return 0;
}