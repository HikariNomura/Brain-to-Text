#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <stdexcept>

struct FoldResult {
    int layer;
    int channel;
    int fold;
    int train_n;
    int tune_n;
    int test_n;
    double mean_r;
    double mean_r2;
};

// Writes every row in one shot. The original Python script's CSV path only
// encoded the word (not layer or channel), and the block that wrote it sat
// inside the per-channel loop, so each write clobbered the previous one --
// only the last channel's 20 rows ever survived on disk. Here `layer` and
// `channel` are included as columns and every row for the word is collected
// and written exactly once, so nothing is silently discarded anymore.
inline void write_results_csv(const std::string& path, const std::vector<FoldResult>& rows) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output file '" + path + "'");
    f << "layer,channel,fold,train_n,tune_n,test_n,mean_r,mean_r2\n";
    f << std::setprecision(17);
    for (const auto& r : rows) {
        f << r.layer << ',' << r.channel << ',' << r.fold << ',' << r.train_n << ',' << r.tune_n << ','
          << r.test_n << ',' << r.mean_r << ',' << r.mean_r2 << '\n';
    }
}
