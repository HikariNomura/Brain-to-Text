#pragma once
// Linear interpolation matching
// scipy.interpolate.interp1d(x, Y, axis=0, kind="linear", bounds_error=False,
// fill_value="extrapolate")
#include <Eigen/Dense>
#include <vector>
#include <algorithm>

// x_known: strictly increasing, length T. Y_known: T x F.
// x_query: length Q. Returns Q x F, linearly interpolated/extrapolated.
inline Eigen::MatrixXd linear_interp_extrapolate(
    const std::vector<double>& x_known,
    const Eigen::MatrixXd& Y_known,
    const std::vector<double>& x_query) {
    int T = (int)x_known.size();
    int F = (int)Y_known.cols();
    int Q = (int)x_query.size();
    Eigen::MatrixXd out(Q, F);

    for (int qi = 0; qi < Q; qi++) {
        double q = x_query[qi];
        // Find segment index i such that x_known[i] <= q <= x_known[i+1],
        // clamped to [0, T-2] so points outside the range extrapolate along
        // the nearest boundary segment (matches fill_value="extrapolate").
        int i;
        if (T < 2) {
            i = 0;
        } else if (q <= x_known[0]) {
            i = 0;
        } else if (q >= x_known[T - 1]) {
            i = T - 2;
        } else {
            auto it = std::upper_bound(x_known.begin(), x_known.end(), q);
            i = (int)(it - x_known.begin()) - 1;
            if (i < 0) i = 0;
            if (i > T - 2) i = T - 2;
        }
        double x0 = x_known[i], x1 = x_known[i + 1];
        double t = (x1 != x0) ? (q - x0) / (x1 - x0) : 0.0;
        out.row(qi) = Y_known.row(i) + t * (Y_known.row(i + 1) - Y_known.row(i));
    }
    return out;
}
