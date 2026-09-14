#pragma once
// Pearson correlation and R^2, matching scipy.stats.pearsonr(...)[0] and
// sklearn.metrics.r2_score(y_true, y_pred).
#include <Eigen/Dense>
#include <cmath>

inline double pearson_r(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    double ma = a.mean(), mb = b.mean();
    Eigen::VectorXd ac = a.array() - ma;
    Eigen::VectorXd bc = b.array() - mb;
    double num = ac.dot(bc);
    double den = std::sqrt(ac.squaredNorm() * bc.squaredNorm());
    return den != 0.0 ? num / den : std::numeric_limits<double>::quiet_NaN();
}

inline double r2_score(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    double mean_true = y_true.mean();
    double ss_res = (y_true - y_pred).squaredNorm();
    double ss_tot = (y_true.array() - mean_true).matrix().squaredNorm();
    return ss_tot != 0.0 ? 1.0 - ss_res / ss_tot : std::numeric_limits<double>::quiet_NaN();
}
