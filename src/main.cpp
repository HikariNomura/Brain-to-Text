// C++ port of mTRF pipeline: 5-fold (train80/tune5/test15), regularization
// tuning via tune data, test evaluated once per fold (average-then-evaluate),
// and fold-to-fold variability (std) recorded alongside the mean.
//
// ラグ範囲の検証版:
//   - ラグ上限 tmax を複数(既定: 0.2s と 0.5s)同じ条件で回し、相関が上がるか比較する
//   - 各 layer × channel について「TRF重みが最大となるラグ(ms)」を peak_lag_ms として出力する
//       重みの大きさ = 各ラグの10個のPCA成分の重みのL2ノルム → fold間で平均 → argmax
//   - ラグプロファイル全体 (layer, channel, lag) も npy で保存する
//   - 比較用に、layerごとのchannel平均r・ピークラグ中央値・tmax間の差を compare_*.csv に出力する
//
// 高速化:
//   - ラグを lag_step(ms) 刻みに間引く (刺激は50Hz補間なので10ms刻みで情報はほぼ落ちない)
//   - 検証用に layers / 被験者 / 単語 を絞れる (layers_to_run, subjects_to_run, words_to_run)
//   - EEGロード・fold平均・Whisper読込/PCA/補間は tmax によらず共通なので1回だけ実行
//   - reg_candidates は前回csvの mean_reg を見て絞ると更に速くなる
//
// 設計方針:
//   - 条件(word_d, word_u など)ごとに独立して5-fold分割(train80/tune5/test15)
//   - trainは80試行平均→1回だけ学習
//   - tuneは5試行平均→reg_candidatesから最良の正則化を選択(foldごとに選び直す)
//   - testは15試行を"先に平均してから"1回だけ評価
//   - 5fold分のr, r2の「平均」と「標準偏差」の両方を記録する
//   - Python版とのbit-exact再現性は目的としない
//
// 注意: trf.hpp の実装に依存する箇所が2つある (LAG_MAJOR と 飛び飛びラグの受け付け)。下記参照。
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
#include "pca.hpp"
#include "interp.hpp"
#include "trf.hpp"
#include "stats.hpp"
#include "butterworth.hpp"
#include "csv.hpp"

namespace fs = std::filesystem;

namespace {

// ================= configuration =================
const std::vector<std::string> hikensya = {"kk", "tr", "ks"};
const std::vector<std::string> imi = {"_a", "_n"};
const int imi_choice = 0;

const std::vector<std::string> word_list = {"ame", "nizi", "momo", "hasi"};
const std::vector<std::string> oto = {"_d", "_u"};

// --- 検証用に絞る (全部回すときは {0,1,2} / {0,1,2,3} / 0..31 にする) ---
const std::vector<int> subjects_to_run = {0, 1, 2};
const std::vector<int> words_to_run    = {0, 1, 2, 3};
const std::vector<int> layers_to_run   = {0, 8, 16, 24, 31};

// --- ラグ設定 ---
const std::vector<double> tmax_list = {0.1, 0.125, 0.15,0.175}; // 比較するラグ上限[s]
const double tmin = 0.0;
const int lag_step = 10; // [サンプル] = [ms] (eeg_sr=1000のとき)。1にすると従来通りの1ms刻み

// trf.hpp の solve_weights が返す重みの並びに合わせること:
//   true : index = lag_idx * n_feat + feat
//   false: index = feat * n_lags + lag_idx
constexpr bool LAG_MAJOR = true;

// 重みの行数が n_lags*n_feat + 1 のとき、余分な1行(バイアス項)が
//   true : 先頭行  (MATLAB mTRF-Toolbox系はこの並び)
//   false: 末尾行
// trf.hpp の solve_weights / build_trf_context で、定数列を設計行列のどこに置いているか確認すること
constexpr bool BIAS_FIRST = true;

const bool EEG_lowpass = false;
const bool EEG_bandpass = false;
const std::string today = "1009_lagcheck"; // 出力フォルダ名 (既存の0913結果を上書きしない)

const double audio_length_sec = 3.0;
const int eeg_sr = 1000;
const int whisper_sr = 50;

const int num_layers_total = 32; // layer 0..31
const int num_channels = 21;
const int num_samples = eeg_sr;  // 1000

// ---- fold構成 & 正則化候補 ----
const int TEST_N = 15;
const int TUNE_N = 5;
const std::vector<double> reg_candidates = {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};

// ================= helpers =================
struct FoldSizing {
    int chunk_size;
    int test_n;
    int tune_n;
    int n_folds;
};

// n_trialsの約数の中から、target_chunk_sizeに一番近いものを選ぶ。
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

Eigen::MatrixXd npy2d_to_matrix(const NpyArray& a) {
    if (a.ndim() != 2) throw std::runtime_error("expected a 2-D array");
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(
        a.data.data(), (Eigen::Index)a.shape[0], (Eigen::Index)a.shape[1]);
    return m;
}

// 指定条件の全チャネルを一括抽出する．
// 戻り値: (n_trials, n_channels * n_time) 行列．
// 列レイアウト: [ch0_t0..ch0_tT-1, ch1_t0..ch1_tT-1, ...]
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

NpyArray apply_filter_all_series(const NpyArray& eeg, const std::vector<double>& b,
                                 const std::vector<double>& a) {
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

NpyArray apply_eeg_lowpass(const NpyArray& eeg, int sr) {
    auto [b, a] = butterworth::design_lowpass(4, 20.0 / (sr / 2.0));
    return apply_filter_all_series(eeg, b, a);
}

NpyArray apply_eeg_bandpass(const NpyArray& eeg, int sr) {
    double nyquist = sr / 2.0;
    auto [b, a] = butterworth::design_bandpass(3, 0.5 / nyquist, 15.0 / nyquist);
    return apply_filter_all_series(eeg, b, a);
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

double median_of(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ラグ(サンプル)列を作る。tmin〜tmax を lag_step 刻みで。
std::vector<int> make_lags(double tmax) {
    std::vector<int> lags;
    for (int l = (int)std::floor(tmin * eeg_sr); l <= (int)std::ceil(tmax * eeg_sr); l += lag_step)
        lags.push_back(l);
    return lags;
}

// 重み w (n_lags*n_feat, 1) からラグごとの重みの大きさ(特徴次元方向のL2ノルム)を返す
Eigen::VectorXd lag_profile(const Eigen::MatrixXd& w, int n_lags, int n_feat) {
    const Eigen::Index n_body = (Eigen::Index)n_lags * n_feat;
    Eigen::Index offset = 0; // バイアス行を飛ばすための先頭オフセット
    if (w.rows() == n_body) {
        offset = 0;
    } else if (w.rows() == n_body + 1) {
        offset = BIAS_FIRST ? 1 : 0; // バイアス行が1行ある場合 (先頭 or 末尾)
    } else {
        throw std::runtime_error("lag_profile: unexpected weight shape (rows=" + std::to_string(w.rows()) +
                                 ", expected " + std::to_string(n_body) + " or " +
                                 std::to_string(n_body + 1) + ")");
    }
    Eigen::VectorXd prof(n_lags);
    for (int l = 0; l < n_lags; l++) {
        double acc = 0.0;
        for (int f = 0; f < n_feat; f++) {
            Eigen::Index idx = offset + (LAG_MAJOR ? (Eigen::Index)l * n_feat + f
                                                   : (Eigen::Index)f * n_lags + l);
            acc += w(idx, 0) * w(idx, 0);
        }
        prof(l) = std::sqrt(acc);
    }
    return prof;
}

// layer・channelごとに、fold間の平均・標準偏差・選ばれた正則化の平均・ピークラグを1行として記録する
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
    double mean_reg;     // foldごとに選ばれたregの算術平均(参考)
    double peak_lag_ms;  // TRF重みが最大のラグ[ms]
};

void write_agg_csv(const std::string& path, const std::vector<AggResult>& rows) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << "layer,channel,n_folds,train_n,tune_n,test_n,mean_r,std_r,mean_r2,std_r2,mean_reg,peak_lag_ms\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        f << r.layer << "," << r.channel << "," << r.n_folds << "," << r.train_n << ","
          << r.tune_n << "," << r.test_n << "," << r.mean_r << "," << r.std_r << ","
          << r.mean_r2 << "," << r.std_r2 << "," << r.mean_reg << "," << r.peak_lag_ms << "\n";
    }
}

// tmaxごとの実行状態 (結果・出力配列・TRFコンテキスト)
struct RunState {
    double tmax = 0.0;
    int tmax_ms = 0;
    std::vector<int> lags;
    std::vector<AggResult> results;
    NpyArray pred, test, prof;
    std::vector<TrfContext> ctx;
};

NpyArray make_zero_array(size_t a, size_t b, size_t c) {
    NpyArray arr;
    arr.shape = {a, b, c};
    arr.data.resize(a * b * c, 0.0);
    return arr;
}

std::string run_dir(const RunState& run, const std::string& subj) {
    return "../data/" + today + "_tmax" + std::to_string(run.tmax_ms) + "/" + subj;
}

// layerごとのchannel平均r・ピークラグ中央値を tmax 間で並べる比較用csv
void write_compare_csv(const std::string& path, const std::vector<RunState>& runs) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    f << "layer";
    for (const auto& run : runs)
        f << ",mean_r_tmax" << run.tmax_ms << ",median_peak_lag_ms_tmax" << run.tmax_ms;
    if (runs.size() >= 2)
        f << ",delta_r_tmax" << runs.back().tmax_ms << "_minus_tmax" << runs.front().tmax_ms;
    f << "\n";

    for (int layer : layers_to_run) {
        f << layer;
        std::vector<double> r_avg;
        for (const auto& run : runs) {
            std::vector<double> rs, peaks;
            for (const auto& row : run.results) {
                if (row.layer != layer) continue;
                rs.push_back(row.mean_r);
                peaks.push_back(row.peak_lag_ms);
            }
            double r = rs.empty() ? std::numeric_limits<double>::quiet_NaN() : mean_of(rs);
            r_avg.push_back(r);
            f << "," << r << "," << median_of(peaks);
        }
        if (runs.size() >= 2) f << "," << (r_avg.back() - r_avg.front());
        f << "\n";
    }
}

struct FoldMeans {
    Eigen::RowVectorXd train;
    Eigen::RowVectorXd tune;
    Eigen::RowVectorXd test;
};

} // namespace

int main() {
    std::ios::sync_with_stdio(false);

    // eeg_time / whisper_time は定数 → 1回だけ構築
    std::vector<double> eeg_time(eeg_sr);
    for (int t = 0; t < eeg_sr; t++) eeg_time[t] = (double)t / eeg_sr;

    const int valid_frames = (int)(audio_length_sec * whisper_sr); // 150
    std::vector<double> whisper_time(valid_frames);
    for (int t = 0; t < valid_frames; t++) whisper_time[t] = (double)t / whisper_sr;

    const auto condition_map = make_condition_map();

    const char* eeg_root_env     = std::getenv("MTRF_EEG_ROOT");
    const char* whisper_root_env = std::getenv("MTRF_WHISPER_ROOT");
    const std::string eeg_root =
        eeg_root_env ? eeg_root_env : "C:/Users/Kurisu/Desktop/Working2024/NPY/03_PreProCutEEG";
    const std::string whisper_root =
        whisper_root_env ? whisper_root_env : "C:/Users/Kurisu/Desktop/R8-hikari/Whisper/Feature_extraction";

    for (int k : subjects_to_run) {
        const std::string subj = hikensya.at(k);
        const std::string eeg_path = eeg_root + "/" + subj + imi[imi_choice] + ".npy";

        std::cout << "Loading EEG: " << eeg_path << "\n";
        NpyArray eeg = load_npy(eeg_path);
        NpyArray eeg_filtered;
        if (EEG_bandpass) {
            eeg_filtered = apply_eeg_bandpass(eeg, eeg_sr);
        } else if (EEG_lowpass) {
            eeg_filtered = apply_eeg_lowpass(eeg, eeg_sr);
        }
        const NpyArray& eeg_used = (EEG_bandpass || EEG_lowpass) ? eeg_filtered : eeg;

        if (eeg_used.ndim() != 4 || (int)eeg_used.shape[2] != num_channels)
            throw std::runtime_error("EEG channel count != " + std::to_string(num_channels));

        // sizing / splits は n_trials にのみ依存 → 被験者ごとに1回だけ計算
        const int n_trials = (int)eeg_used.shape[0];
        const FoldSizing sizing = choose_fold_sizing(n_trials, TEST_N, TUNE_N);
        std::cout << "Fold sizing: chunk=" << sizing.chunk_size
                  << " (test=" << sizing.test_n << ", tune=" << sizing.tune_n << ")"
                  << ", n_folds=" << sizing.n_folds << "\n";
        const std::vector<FoldSplit> splits = make_fold_splits(n_trials, sizing.test_n, sizing.tune_n);
        const int train_n = n_trials - sizing.chunk_size;

        for (int wi : words_to_run) {
            for (int oi = 0; oi < (int)oto.size(); oi++) {
                const std::string word = word_list.at(wi) + oto[oi];
                std::cout << std::string(60, '=') << "\n" << subj << " / " << word << "\n"
                          << std::string(60, '=') << "\n";
                const int condition_idx = condition_map.at(word);

                // 全チャネルを1回まとめて抽出: (n_trials, n_channels * n_time)
                Eigen::MatrixXd Y_all = extract_condition_all_channels(eeg_used, condition_idx, eeg_sr);

                // foldごとの train/tune/test 平均を全チャネル同時に事前計算 (layer/tmaxによらず不変)
                std::vector<FoldMeans> fold_means(splits.size());
                {
                    const int ncT = num_channels * num_samples;
                    for (size_t fold = 0; fold < splits.size(); fold++) {
                        const FoldSplit& sp = splits[fold];
                        FoldMeans& fm = fold_means[fold];
                        fm.train = Eigen::RowVectorXd::Zero(ncT);
                        fm.tune  = Eigen::RowVectorXd::Zero(ncT);
                        fm.test  = Eigen::RowVectorXd::Zero(ncT);
                        for (int t : sp.train_trials) fm.train += Y_all.row(t);
                        fm.train /= (double)sp.train_trials.size();
                        for (int t : sp.tune_trials)  fm.tune  += Y_all.row(t);
                        fm.tune  /= (double)sp.tune_trials.size();
                        for (int t : sp.test_trials)  fm.test  += Y_all.row(t);
                        fm.test  /= (double)sp.test_trials.size();
                    }
                }

                // tmaxごとの実行状態を初期化
                std::vector<RunState> runs(tmax_list.size());
                for (size_t ri = 0; ri < runs.size(); ri++) {
                    RunState& run = runs[ri];
                    run.tmax = tmax_list[ri];
                    run.tmax_ms = (int)std::lround(run.tmax * 1000.0);
                    run.lags = make_lags(run.tmax);
                    run.pred = make_zero_array(num_layers_total, num_channels, num_samples);
                    run.test = make_zero_array(num_layers_total, num_channels, num_samples);
                    run.prof = make_zero_array(num_layers_total, num_channels, run.lags.size());
                    std::cout << "tmax=" << run.tmax_ms << "ms: n_lags=" << run.lags.size()
                              << " (step " << lag_step << ")\n";
                }

                for (int layer : layers_to_run) {
                    std::cout << "\nLayer " << layer << "\n";

                    // Whisper読込・PCA・補間は tmax によらず共通 → layerごとに1回だけ
                    std::string whisper_path = whisper_root + "/" + word + "/layer_" + std::to_string(layer) + ".npy";
                    NpyArray whisper_npy = load_npy(whisper_path);
                    whisper_npy.squeeze();
                    Eigen::MatrixXd whisper_full = npy2d_to_matrix(whisper_npy);

                    if (whisper_full.rows() < valid_frames)
                        throw std::runtime_error("whisper array shorter than audio_length_sec*whisper_sr");
                    Eigen::MatrixXd whisper = whisper_full.topRows(valid_frames);

                    Eigen::MatrixXd whisper_pca = pca_fit_transform(whisper, 10);
                    Eigen::MatrixXd X = linear_interp_extrapolate(whisper_time, whisper_pca, eeg_time);
                    std::cout << "Stimulus : (" << X.rows() << ", " << X.cols() << ")\n";
                    const int n_feat = (int)X.cols();

                    for (RunState& run : runs) {
                        const int n_lags = (int)run.lags.size();
                        std::cout << "-- tmax=" << run.tmax_ms << "ms --\n";

                        run.ctx.clear();
                        run.ctx.reserve(reg_candidates.size());
                        for (double reg : reg_candidates)
                            run.ctx.push_back(build_trf_context(X, run.lags, reg, eeg_sr));

                        for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) {
                            std::vector<double> fold_r, fold_r2, fold_reg;
                            fold_r.reserve(splits.size());
                            fold_r2.reserve(splits.size());
                            fold_reg.reserve(splits.size());

                            Eigen::VectorXd ensemble_y_pred = Eigen::VectorXd::Zero(eeg_sr);
                            Eigen::VectorXd ensemble_y_test = Eigen::VectorXd::Zero(eeg_sr);
                            Eigen::VectorXd sum_profile = Eigen::VectorXd::Zero(n_lags);

                            const int seg_start = channel_idx * num_samples;

                            for (size_t fold = 0; fold < splits.size(); fold++) {
                                Eigen::VectorXd mean_y_train =
                                    fold_means[fold].train.segment(seg_start, num_samples).transpose();
                                Eigen::VectorXd mean_y_tune =
                                    fold_means[fold].tune.segment(seg_start, num_samples).transpose();
                                Eigen::VectorXd mean_y_test =
                                    fold_means[fold].test.segment(seg_start, num_samples).transpose();

                                // tune: 候補regごとに重みと予測をキャッシュしながら最良regを選ぶ
                                int best_idx = -1;
                                double best_tune_r = -std::numeric_limits<double>::infinity();
                                std::vector<Eigen::MatrixXd> w_cache(reg_candidates.size());
                                std::vector<Eigen::VectorXd> y_pred_cache(reg_candidates.size());
                                for (size_t ci = 0; ci < reg_candidates.size(); ci++) {
                                    w_cache[ci] = run.ctx[ci].solve_weights(mean_y_train);
                                    y_pred_cache[ci] = run.ctx[ci].predict(w_cache[ci]).col(0);
                                    double tune_r = pearson_r(mean_y_tune, y_pred_cache[ci]);
                                    if (tune_r > best_tune_r) { best_tune_r = tune_r; best_idx = (int)ci; }
                                }
                                const Eigen::VectorXd& y_pred = y_pred_cache[best_idx];

                                // test: 事前計算済み mean_y_test で1回だけ評価
                                double r  = pearson_r(mean_y_test, y_pred);
                                double r2 = r2_score(mean_y_test, y_pred);

                                fold_r.push_back(r);
                                fold_r2.push_back(r2);
                                fold_reg.push_back(reg_candidates[best_idx]);

                                ensemble_y_pred += y_pred;
                                ensemble_y_test += mean_y_test;

                                // 選ばれたregの重みからラグプロファイルを加算
                                sum_profile += lag_profile(w_cache[best_idx], n_lags, n_feat);
                            }

                            const double n_folds_d = (double)splits.size();
                            ensemble_y_pred /= n_folds_d;
                            ensemble_y_test /= n_folds_d;
                            sum_profile /= n_folds_d;

                            Eigen::Index peak = 0;
                            sum_profile.maxCoeff(&peak);
                            const double peak_lag_ms = run.lags[(size_t)peak] * 1000.0 / eeg_sr;

                            size_t base_idx = (size_t)layer * (num_channels * num_samples) +
                                              (size_t)channel_idx * num_samples;
                            for (int t = 0; t < num_samples; t++) {
                                run.pred.data[base_idx + t] = ensemble_y_pred(t);
                                run.test.data[base_idx + t] = ensemble_y_test(t);
                            }
                            size_t prof_base = ((size_t)layer * num_channels + (size_t)channel_idx) * n_lags;
                            for (int l = 0; l < n_lags; l++) run.prof.data[prof_base + l] = sum_profile(l);

                            double mr   = mean_of(fold_r);
                            double sr   = std_of(fold_r, mr);
                            double mr2  = mean_of(fold_r2);
                            double sr2  = std_of(fold_r2, mr2);
                            double mreg = mean_of(fold_reg);

                            std::cout << std::fixed << std::setprecision(4)
                                      << "  ch" << channel_idx << ": r=" << mr << " (std=" << sr << "), "
                                      << "r2=" << mr2 << " (std=" << sr2 << "), "
                                      << "peak_lag=" << std::setprecision(0) << peak_lag_ms << "ms\n";

                            run.results.push_back(AggResult{
                                layer, channel_idx, (int)splits.size(),
                                train_n, sizing.tune_n, sizing.test_n,
                                mr, sr, mr2, sr2, mreg, peak_lag_ms
                            });
                        }
                    }
                }

                // ---- 出力 ----
                for (const RunState& run : runs) {
                    const std::string dir = run_dir(run, subj);
                    fs::create_directories(dir);

                    std::string agg_path = dir + "/mtrf_agg_" + word + ".csv";
                    write_agg_csv(agg_path, run.results);
                    std::cout << "\nSaved: " << agg_path << " (" << run.results.size() << " rows)\n";

                    std::string pred_path = dir + "/mtrf_pred_" + word + ".npy";
                    std::string test_path = dir + "/mtrf_test_" + word + ".npy";
                    std::string prof_path = dir + "/mtrf_lagprofile_" + word + ".npy";
                    save_npy(pred_path, run.pred);
                    save_npy(test_path, run.test);
                    save_npy(prof_path, run.prof);
                    std::cout << "Saved: " << pred_path << " & " << test_path << " & " << prof_path << "\n";
                }

                std::string cmp_path = "../data/" + today + "/" + subj + "/compare_" + word + ".csv";
                write_compare_csv(cmp_path, runs);
                std::cout << "Saved: " << cmp_path << "\n";
            }
        }
    }

    return 0;
}