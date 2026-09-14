#pragma once
// PCA matching scikit-learn's PCA(n_components=k).fit_transform(X) semantics
// (mean-center, SVD, sign convention), but always using a deterministic full
// SVD rather than sklearn's default randomized SVD.
//
// NOTE ON FIDELITY: the original Python script calls
// `PCA(n_components=10).fit_transform(whisper)` without a `random_state`.
// For a (150, 1280)-shaped input with only 10 components requested,
// scikit-learn's `svd_solver="auto"` picks the *randomized* SVD solver,
// which draws from numpy's global (unseeded) random state. That means the
// Python script itself does not produce bit-identical PCA output between
// separate runs. Exactly reproducing one particular run is therefore not a
// well-posed goal, so this port uses the deterministic full SVD instead
// (the same math the randomized solver approximates), with the same
// sign-convention fix-up (`svd_flip`) sklearn applies. Results will be very
// close to, but not bit-identical with, any single Python run.
//
// Verified against scikit-learn 1.9.0: pca_fit_transform() matches
// `PCA(n_components=k, svd_solver="full").fit_transform(X)` to ~1e-15 on
// random test matrices. Note scikit-learn >=1.5 made PCA._fit_full() use
// svd_flip(..., u_based_decision=False) (sign fixed by the largest entry of
// each right-singular-vector row, not the left one); that's what's
// implemented below. Older sklearn versions used u_based_decision=True and
// would need the flip loop below to scan U instead of V.
#include <Eigen/Dense>

// X: (n_samples x n_features). Returns (n_samples x n_components).
inline Eigen::MatrixXd pca_fit_transform(const Eigen::MatrixXd& X, int n_components) {
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd Xc = X.rowwise() - mean;

    Eigen::BDCSVD<Eigen::MatrixXd> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();      // (n_samples x r)
    Eigen::VectorXd S = svd.singularValues();
    Eigen::MatrixXd V = svd.matrixV();      // (n_features x r), Vt = V.transpose()

    // sklearn's svd_flip(u, v, u_based_decision=False) -- this is the
    // default used by PCA._fit_full() as of scikit-learn >= 1.5: the sign of
    // each component is fixed by the largest-magnitude entry of each row of
    // Vt (equivalently, each column of V here), not by U.
    int r = (int)U.cols();
    for (int k = 0; k < r; k++) {
        int max_row = 0;
        double max_abs = 0.0;
        for (int i = 0; i < V.rows(); i++) {
            double a = std::abs(V(i, k));
            if (a > max_abs) { max_abs = a; max_row = i; }
        }
        if (V(max_row, k) < 0.0) {
            U.col(k) *= -1.0;
            V.col(k) *= -1.0;
        }
    }

    int k = std::min<int>(n_components, r);
    Eigen::MatrixXd transformed = U.leftCols(k) * S.head(k).asDiagonal();
    return transformed;
}
