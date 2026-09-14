#pragma once
// Forward (direction=1) ridge-regression TRF model, matching the relevant
// subset of the `mtrf` PyPI package (powerfulbean/mTRFpy)'s TRF class:
// lag_matrix() + covariance_matrices() + fit_weights_with_covariance_matrices()
// with method="ridge" and a scalar regularization value (i.e. the path taken
// by `TRF(direction=1).train(..., regularization=<float>)`, which skips the
// cross-validation branch entirely since a single float isn't a list).
//
// KEY ALGEBRAIC SIMPLIFICATION (see README): in the source script every
// trial of a given word/layer shares the exact same stimulus matrix X (only
// the EEG response Y differs per trial). Because covariance_matrices()
// averages `x_lag.T @ x_lag` and `x_lag.T @ y_i` over trials, and x_lag is
// identical across trials, this reduces exactly (in real arithmetic) to:
//   cov_xx = x_lag.T @ x_lag                         (trial count cancels)
//   cov_xy = x_lag.T @ mean_over_train_trials(y_i)
// This lets a single lag matrix / covariance / factorization be reused
// across every channel and fold for a given word+layer, which is what makes
// this port tractable (a literal per-trial port would need ~1680 separate
// 2011x2011 covariance accumulations and is not practical to run).
//
// It also means fs-scaling applied during "training" in the reference
// implementation (`weight_matrix = solve(...) / (1/fs)`) is exactly undone
// again during "prediction" (`w = weights_reshaped * (1/fs)`), so the model
// weight actually used for prediction is simply `solve(cov_xx + regmat, cov_xy)`
// with no fs scaling at all -- that's what `TrfContext::solve_weights` below
// returns.
#include <Eigen/Dense>
#include <vector>

// Builds the time-lagged, zero-padded design matrix with a bias column
// prepended, matching mtrf.matrices.lag_matrix(x, lags, zeropad=True, bias=True).
// Column layout: [bias, lag0*F features, lag1*F features, ...].
inline Eigen::MatrixXd build_lag_matrix(const Eigen::MatrixXd& X, const std::vector<int>& lags) {
    int n_samples = (int)X.rows();
    int n_variables = (int)X.cols();
    int n_lags = (int)lags.size();

    Eigen::MatrixXd x_lag = Eigen::MatrixXd::Zero(n_samples, n_variables * n_lags);
    for (int idx = 0; idx < n_lags; idx++) {
        int lag = lags[idx];
        int col0 = idx * n_variables;
        if (lag < 0) {
            int rows = n_samples + lag; // lag negative
            if (rows > 0) x_lag.block(0, col0, rows, n_variables) = X.block(-lag, 0, rows, n_variables);
        } else if (lag > 0) {
            int rows = n_samples - lag;
            if (rows > 0) x_lag.block(lag, col0, rows, n_variables) = X.block(0, 0, rows, n_variables);
        } else {
            x_lag.block(0, col0, n_samples, n_variables) = X;
        }
    }

    Eigen::MatrixXd with_bias(n_samples, 1 + n_variables * n_lags);
    with_bias.col(0).setOnes();
    with_bias.rightCols(n_variables * n_lags) = x_lag;
    return with_bias;
}

// Precomputed, channel/fold-independent pieces of a forward TRF model for one
// (word, layer) stimulus matrix X and a fixed regularization/fs/lags setup.
struct TrfContext {
    Eigen::MatrixXd x_lag;              // T x S  (S = 1 + n_features * n_lags)
    Eigen::LDLT<Eigen::MatrixXd> ldlt;   // factorization of (cov_xx + regmat)

    // Solve for the model weights given a right-hand side y (T-length
    // response, or T x n mean responses batched as columns).
    Eigen::MatrixXd solve_weights(const Eigen::MatrixXd& y) const {
        Eigen::MatrixXd cov_xy = x_lag.transpose() * y; // S x n
        return ldlt.solve(cov_xy);                       // S x n
    }

    Eigen::MatrixXd predict(const Eigen::MatrixXd& w) const {
        return x_lag * w; // T x n
    }
};

inline TrfContext build_trf_context(const Eigen::MatrixXd& X, const std::vector<int>& lags,
                                     double regularization, double fs) {
    TrfContext ctx;
    ctx.x_lag = build_lag_matrix(X, lags);
    Eigen::MatrixXd cov_xx = ctx.x_lag.transpose() * ctx.x_lag;
    // ridge regularization_matrix: identity with [0,0]=0, scaled by
    // regularization / (1/fs) == regularization * fs.
    double lambda = regularization * fs;
    for (int i = 1; i < cov_xx.rows(); i++) cov_xx(i, i) += lambda;
    ctx.ldlt.compute(cov_xx);
    return ctx;
}
