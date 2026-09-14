// C++ port of mTRF pipeline: 5-fold (train80/tune5/test15), regularization
// tuning via tune data, test evaluated once per fold (average-then-evaluate),
// and fold-to-fold variability (std) recorded alongside the mean.
//
// 設計方針:
//   - 条件(word_d, word_u など)ごとに独立して5-fold分割(train80/tune5/test15)
//   - trainは80試行平均→1回だけ学習(ノイズ除去が目的なので、可能な限り多くの試行を平均)
//   - tuneは5試行平均→reg_candidatesから最良の正則化を選択(foldごとに選び直す)
//   - testは15試行を"先に平均してから"1回だけ評価(individual評価はしない)
//   - 5fold分のr, r2の「平均」と「標準偏差」の両方を記録する
//   - Python版とのbit-exact再現性は目的としない
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <numeric>
#include <limits>
#include <cmath>
#include <fstream>

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

// ---- configuration (mirrors the Python script's constants) ----
const std::vector<std::string> hikensya = {"kk", "tr", "ks"};
const std::vector<std::string> imi = {"_a", "_n"};
int hikensya_choice = 0;
const int imi_choice = 0;

const std::vector<std::string> word_list = {"ame", "nizi", "momo", "hasi"};
const std::vector<std::string> oto = {"_d", "_u"};
int wordchoice = 0;

const bool EEG_lowpass = false;
const bool EEG_bandpass = false;
const std::string today = "0913";

const double audio_length_sec = 3.0;
const int eeg_sr = 1000;
// const int eeg_samples = 999;
const int whisper_sr = 50;
const double tmin = 0.0;
const double tmax = 0.2;

// ---- fold構成 & 正則化候補 ----
const int TEST_N = 15;
const int TUNE_N = 5;
const std::vector<double> reg_candidates = {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};

// n_trialsの約数の中から、target_chunk_sizeに一番近いものを選ぶ。
// (n_trialsが100と異なる被験者がいても自動で対応するため)
struct FoldSizing {
    int chunk_size;
    int test_n;
    int tune_n;
    int n_folds;
};

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

    // target比率(test:tune)を維持しつつ、選ばれたchunk_sizeに合わせて配分し直す
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

Eigen::MatrixXd extract_condition_channel(const NpyArray& eeg, int condition_idx, int channel_idx,
                                           int n_time) {
    if (eeg.ndim() != 4) throw std::runtime_error("expected eeg to be a 4-D array");
    size_t n_trials = eeg.shape[0];
    size_t n_cond = eeg.shape[1];
    size_t n_chan = eeg.shape[2];
    size_t T = eeg.shape[3];
    if ((size_t)condition_idx >= n_cond || (size_t)channel_idx >= n_chan || (size_t)n_time > T) {
        throw std::runtime_error("extract_condition_channel: index out of range");
    }
    auto strides = eeg.strides();
    Eigen::MatrixXd Y((Eigen::Index)n_trials, n_time);
    for (size_t trial = 0; trial < n_trials; trial++) {
        size_t base = trial * strides[0] + (size_t)condition_idx * strides[1] +
                      (size_t)channel_idx * strides[2];
        for (int t = 0; t < n_time; t++) Y((Eigen::Index)trial, t) = eeg.data[base + (size_t)t];
    }
    return Y;
}

// 指定条件の全チャネルを一括抽出する．
// 戻り値: (n_trials, n_channels * n_time) 行列．
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
    auto strides = eeg.strides();
    size_t n_series = eeg.data.size() / T;
    std::vector<double> buf(T);
    for (size_t s = 0; s < n_series; s++) {
        size_t base = s * T;
        for (size_t t = 0; t < T; t++) buf[t] = eeg.data[base + t];
        std::vector<double> filtered = butterworth::filtfilt(b, a, buf);
        for (size_t t = 0; t < T; t++) out.data[base + t] = filtered[t];
    }
    (void)strides;
    return out;
}

NpyArray apply_eeg_bandpass(const NpyArray& eeg, int sr) {
    double nyquist = sr / 2.0;
    auto [b, a] = butterworth::design_bandpass(3, 0.5 / nyquist, 15.0 / nyquist);
    NpyArray out = eeg;
    size_t T = eeg.shape[3];
    auto strides = eeg.strides();
    size_t n_series = eeg.data.size() / T;
    std::vector<double> buf(T);
    for (size_t s = 0; s < n_series; s++) {
        size_t base = s * T;
        for (size_t t = 0; t < T; t++) buf[t] = eeg.data[base + t];
        std::vector<double> filtered = butterworth::filtfilt(b, a, buf);
        for (size_t t = 0; t < T; t++) out.data[base + t] = filtered[t];
    }
    (void)strides;
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

Eigen::VectorXd average_rows(const Eigen::MatrixXd& Y, const std::vector<int>& trials) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(Y.cols());
    for (int t : trials) v += Y.row(t).transpose();
    v /= (double)trials.size();
    return v;
}

double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

double std_of(const std::vector<double>& v, double mean) {
    double acc = 0.0;
    for (double x : v) acc += (x - mean) * (x - mean);
    return std::sqrt(acc / (double)v.size());
}

// レイヤー・チャンネルごとに、fold間の平均・標準偏差・選ばれた正則化の平均を1行として記録する
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
    double mean_reg;   // 参考: foldごとに選ばれたregの平均(対数平均の方が意味的には適切だが、まずは算術平均)
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
    // [最適化7] cout の stdio 同期を無効化してログ出力の I/O コストを削減
    std::ios::sync_with_stdio(false);

    // [最適化5] lags は tmin/tmax/eeg_sr のみに依存 → 全ループ外で1回だけ構築
    std::vector<int> lags;
    for (int l = (int)std::floor(tmin * eeg_sr); l <= (int)std::ceil(tmax * eeg_sr); l++)
        lags.push_back(l);

    // [最適化6] eeg_time / whisper_time はいずれも定数 → 全ループ外で1回だけ構築
    std::vector<double> eeg_time(eeg_sr);
    for (int t = 0; t < eeg_sr; t++) eeg_time[t] = (double)t / eeg_sr;

    const int valid_frames = (int)(audio_length_sec * whisper_sr); // 150
    std::vector<double> whisper_time(valid_frames);
    for (int t = 0; t < valid_frames; t++) whisper_time[t] = (double)t / whisper_sr;

    for (int k = 0; k < 3; k++) {
        hikensya_choice = k;
    for (int i = 0; i < 4; i++) {
        wordchoice = i;

    auto condition_map = make_condition_map();
    std::vector<std::string> words = {word_list[wordchoice] + oto[0], word_list[wordchoice] + oto[1]};
    int layers = 31;

    const char* eeg_root_env    = std::getenv("MTRF_EEG_ROOT");
    const char* whisper_root_env = std::getenv("MTRF_WHISPER_ROOT");
    const std::string eeg_root =
        eeg_root_env ? eeg_root_env : "C:/Users/Kurisu/Desktop/Working2024/NPY/03_PreProCutEEG";
    const std::string whisper_root =
        whisper_root_env ? whisper_root_env : "C:/Users/Kurisu/Desktop/R8-hikari/Whisper/Feature_extraction";

    const std::string eeg_path = eeg_root + "/" + hikensya[hikensya_choice] + imi[imi_choice] + ".npy";

    std::cout << "Loading EEG: " << eeg_path << "\n";
    NpyArray eeg = load_npy(eeg_path);
    NpyArray eeg_filtered;
    if (EEG_bandpass) {
        eeg_filtered = apply_eeg_bandpass(eeg, eeg_sr);
    } else if (EEG_lowpass) {
        eeg_filtered = apply_eeg_lowpass(eeg, eeg_sr);
    }

    const NpyArray& eeg_used = (EEG_bandpass || EEG_lowpass) ? eeg_filtered : eeg;
    fs::create_directories(today);

    // [最適化4] sizing / splits は n_trials にのみ依存 → EEG ロード後に1回だけ計算
    int n_trials = (int)eeg_used.shape[0];
    FoldSizing sizing = choose_fold_sizing(n_trials, TEST_N, TUNE_N);
    std::cout << "Fold sizing: chunk=" << sizing.chunk_size
              << " (test=" << sizing.test_n << ", tune=" << sizing.tune_n << ")"
              << ", n_folds=" << sizing.n_folds << "\n";
    std::vector<FoldSplit> splits = make_fold_splits(n_trials, sizing.test_n, sizing.tune_n);

    for (const std::string& word : words) {
        std::cout << std::string(60, '=') << "\n" << word << "\n" << std::string(60, '=') << "\n";
        int condition_idx = condition_map.at(word);

        std::vector<AggResult> word_results;

        const int num_layers   = layers + 1; // 32
        const int num_channels = 21;
        const int num_samples  = eeg_sr;     // 1000

        NpyArray pred_array;
        pred_array.shape = {(size_t)num_layers, (size_t)num_channels, (size_t)num_samples};
        pred_array.data.resize(num_layers * num_channels * num_samples, 0.0);

        NpyArray test_array;
        test_array.shape = {(size_t)num_layers, (size_t)num_channels, (size_t)num_samples};
        test_array.data.resize(num_layers * num_channels * num_samples, 0.0);

        // [最適化3] 全チャネルを1回まとめて抽出: shape (n_trials, n_channels * n_time)
        // EEGデータは layer によらず不変なので word ループ内・layer ループ外で計算
        std::cout << "EEG shape[0..3] : (" << eeg_used.shape[0] << ", " << eeg_used.shape[1]
                  << ", " << eeg_used.shape[2] << ", " << eeg_used.shape[3] << ")\n";
        Eigen::MatrixXd Y_all = extract_condition_all_channels(eeg_used, condition_idx, eeg_sr);

        // [最適化3] fold ごとの train/tune/test 平均を全チャネル同時に事前計算
        // fold_means[f].{train,tune,test}: shape (n_channels * n_time,)
        // layer ループをまたいで変わらないため、ここで一度だけ計算する
        struct FoldMeans {
            Eigen::RowVectorXd train;
            Eigen::RowVectorXd tune;
            Eigen::RowVectorXd test;
        };
        std::vector<FoldMeans> fold_means(splits.size());
        {
            int ncT = num_channels * num_samples;
            for (size_t fold = 0; fold < splits.size(); fold++) {
                const FoldSplit& sp = splits[fold];
                fold_means[fold].train = Eigen::RowVectorXd::Zero(ncT);
                fold_means[fold].tune  = Eigen::RowVectorXd::Zero(ncT);
                fold_means[fold].test  = Eigen::RowVectorXd::Zero(ncT);
                for (int t : sp.train_trials) fold_means[fold].train += Y_all.row(t);
                fold_means[fold].train /= (double)sp.train_trials.size();
                for (int t : sp.tune_trials)  fold_means[fold].tune  += Y_all.row(t);
                fold_means[fold].tune  /= (double)sp.tune_trials.size();
                for (int t : sp.test_trials)  fold_means[fold].test  += Y_all.row(t);
                fold_means[fold].test  /= (double)sp.test_trials.size();
            }
        }

        for (int layer = 0; layer <= layers; layer++) {
            std::cout << "\nLayer " << layer << "\n";

            std::string whisper_path = whisper_root + "/" + word + "/layer_" + std::to_string(layer) + ".npy";
            NpyArray whisper_npy = load_npy(whisper_path);
            whisper_npy.squeeze();
            Eigen::MatrixXd whisper_full = npy2d_to_matrix(whisper_npy);

            if (whisper_full.rows() < valid_frames)
                throw std::runtime_error("whisper array shorter than audio_length_sec*whisper_sr");
            Eigen::MatrixXd whisper = whisper_full.topRows(valid_frames);
            std::cout << "Whisper : (" << whisper.rows() << ", " << whisper.cols() << ")\n";

            Eigen::MatrixXd whisper_pca = pca_fit_transform(whisper, 10);
            std::cout << "After PCA : (" << whisper_pca.rows() << ", " << whisper_pca.cols() << ")\n";

            Eigen::MatrixXd X = linear_interp_extrapolate(whisper_time, whisper_pca, eeg_time);
            std::cout << "Stimulus : (" << X.rows() << ", " << X.cols() << ")\n";

            std::vector<TrfContext> ctx_candidates;
            ctx_candidates.reserve(reg_candidates.size());
            for (double reg : reg_candidates)
                ctx_candidates.push_back(build_trf_context(X, lags, reg, eeg_sr));

            for (int channel_idx = 0; channel_idx < 21; channel_idx++) {
                std::cout << std::string(50, '=') << "\nChannel " << channel_idx << "\n"
                          << std::string(50, '=') << "\n";

                std::vector<double> fold_r, fold_r2, fold_reg;
                fold_r.reserve(splits.size());
                fold_r2.reserve(splits.size());
                fold_reg.reserve(splits.size());

                Eigen::VectorXd ensemble_y_pred = Eigen::VectorXd::Zero(eeg_sr);
                Eigen::VectorXd ensemble_y_test = Eigen::VectorXd::Zero(eeg_sr);

                const int seg_start = channel_idx * num_samples;

                for (size_t fold = 0; fold < splits.size(); fold++) {
                    // [最適化3] 事前計算した fold 平均から該当チャネルのセグメントを参照
                    Eigen::VectorXd mean_y_train =
                        fold_means[fold].train.segment(seg_start, num_samples).transpose();
                    Eigen::VectorXd mean_y_tune =
                        fold_means[fold].tune.segment(seg_start, num_samples).transpose();
                    Eigen::VectorXd mean_y_test =
                        fold_means[fold].test.segment(seg_start, num_samples).transpose();

                    // ------------------------------------------
                    // tune: 候補regごとにy_predをキャッシュしながら最良regを選ぶ
                    // [最適化1] 選択後に solve_weights を再実行しない
                    // ------------------------------------------
                    int best_idx = -1;
                    double best_tune_r = -std::numeric_limits<double>::infinity();
                    std::vector<Eigen::VectorXd> y_pred_cache(reg_candidates.size());
                    for (size_t ci = 0; ci < reg_candidates.size(); ci++) {
                        Eigen::MatrixXd w = ctx_candidates[ci].solve_weights(mean_y_train);
                        y_pred_cache[ci]  = ctx_candidates[ci].predict(w).col(0);
                        double tune_r = pearson_r(mean_y_tune, y_pred_cache[ci]);
                        if (tune_r > best_tune_r) { best_tune_r = tune_r; best_idx = (int)ci; }
                    }
                    // 再計算なしでキャッシュ済みの予測を使用
                    const Eigen::VectorXd& y_pred = y_pred_cache[best_idx];

                    // ------------------------------------------
                    // test: 事前計算済み mean_y_test で1回だけ評価
                    // ------------------------------------------
                    double r  = pearson_r(mean_y_test, y_pred);
                    double r2 = r2_score(mean_y_test, y_pred);

                    fold_r.push_back(r);
                    fold_r2.push_back(r2);
                    fold_reg.push_back(reg_candidates[best_idx]);

                    ensemble_y_pred += y_pred;
                    ensemble_y_test += mean_y_test;

                    std::cout << "Fold " << (fold + 1) << " reg=" << reg_candidates[best_idx]
                              << " r=" << std::fixed << std::setprecision(4) << r
                              << " r2=" << r2 << "\n";
                }

                ensemble_y_pred /= (double)splits.size();
                ensemble_y_test /= (double)splits.size();

                size_t base_idx = (size_t)layer * (num_channels * num_samples) +
                                  (size_t)channel_idx * num_samples;
                for (int t = 0; t < num_samples; t++) {
                    pred_array.data[base_idx + t] = ensemble_y_pred(t);
                    test_array.data[base_idx + t] = ensemble_y_test(t);
                }

                double mr   = mean_of(fold_r);
                double sr   = std_of(fold_r, mr);
                double mr2  = mean_of(fold_r2);
                double sr2  = std_of(fold_r2, mr2);
                double mreg = mean_of(fold_reg);

                std::cout << "  => Mean r = " << mr << " (std=" << sr << "), "
                          << "Mean r2 = " << mr2 << " (std=" << sr2 << ")\n";

                word_results.push_back(AggResult{
                    layer, channel_idx, (int)splits.size(),
                    sizing.chunk_size - sizing.tune_n - sizing.test_n, // train_n
                    sizing.tune_n, sizing.test_n,
                    mr, sr, mr2, sr2, mreg
                });
            }
        }

        std::string out_path = "../data/" + today + "/" + hikensya[hikensya_choice] + "/mtrf_agg_" + word + ".csv";
        write_agg_csv(out_path, word_results);
        std::cout << "\nSaved: " << out_path << " (" << word_results.size() << " rows)\n";

        std::string pred_out_path = "../data/" + today + "/" + hikensya[hikensya_choice] + "/mtrf_pred_" + word + ".npy";
        std::string test_out_path = "../data/" + today + "/" + hikensya[hikensya_choice] + "/mtrf_test_" + word + ".npy";
        save_npy(pred_out_path, pred_array);
        save_npy(test_out_path, test_array);
        std::cout << "Saved: " << pred_out_path << " & " << test_out_path << "\n";
    }
    }
    }

    return 0;
}