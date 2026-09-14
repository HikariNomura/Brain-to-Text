#pragma once
// Digital Butterworth low-pass filter design (bilinear transform) and
// zero-phase filtering (filtfilt), matching
// scipy.signal.butter(N, Wn, btype="low") + scipy.signal.filtfilt(b, a, x).
//
// This is only exercised when EEG_lowpass=True in main.cpp, which is NOT the
// case in the original script's default configuration (EEG_lowpass=False),
// so it has no effect on the reference run's output. It's included for
// completeness / in case that flag is ever flipped on. Edge-padding and
// initial-condition handling follow scipy's default filtfilt behavior
// (padtype="odd", padlen=3*max(len(a),len(b)), lfilter_zi initial state),
// but exact bit-parity with scipy is not guaranteed.
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

namespace butterworth {

using Cplx = std::complex<double>;

// Not using M_PI: it's a non-standard extension and isn't reliably defined
// by <cmath> across MSVC/MinGW without extra defines.
constexpr double kPi = 3.14159265358979323846;

inline std::vector<Cplx> poly_from_roots(const std::vector<Cplx>& roots) {
    std::vector<Cplx> c = {Cplx(1.0, 0.0)};
    for (const auto& r : roots) {
        std::vector<Cplx> next(c.size() + 1, Cplx(0.0, 0.0));
        for (size_t i = 0; i < c.size(); i++) {
            next[i] += c[i];
            next[i + 1] -= c[i] * r;
        }
        c = next;
    }
    return c;
}

// Returns {b, a}, each of length N+1, for a low-pass Butterworth filter of
// order N with normalized cutoff Wn in (0, 1), 1 == Nyquist frequency.
inline std::pair<std::vector<double>, std::vector<double>> design_lowpass(int N, double Wn) {
    // Analog Butterworth prototype poles (cutoff = 1 rad/s, gain k=1).
    std::vector<Cplx> p(N);
    for (int k = 0; k < N; k++) {
        double theta = kPi * (2.0 * k + N + 1) / (2.0 * N);
        p[k] = Cplx(std::cos(theta), std::sin(theta));
    }
    double k_lp_gain = 1.0;

    // Pre-warp and scale to the desired cutoff (lp2lp with wo=warped).
    double fs = 2.0; // scipy's internal convention when Wn is pre-normalized
    double warped = 2.0 * fs * std::tan(kPi * Wn / fs);
    for (auto& pk : p) pk *= warped;
    double k_lp = k_lp_gain * std::pow(warped, N); // z empty -> degree = N

    // Bilinear transform to the z-plane.
    double fs2 = 2.0 * fs;
    std::vector<Cplx> p_z(N);
    Cplx denom_prod(1.0, 0.0);
    for (int i = 0; i < N; i++) {
        p_z[i] = (fs2 + p[i]) / (fs2 - p[i]);
        denom_prod *= (fs2 - p[i]);
    }
    std::vector<Cplx> z_z(N, Cplx(-1.0, 0.0)); // all zeros land at z=-1
    Cplx k_z = k_lp * (Cplx(1.0, 0.0) / denom_prod); // prod(fs2 - z) over empty z is 1

    std::vector<Cplx> b_c = poly_from_roots(z_z);
    std::vector<Cplx> a_c = poly_from_roots(p_z);

    std::vector<double> b(N + 1), a(N + 1);
    for (int i = 0; i <= N; i++) {
        b[i] = (k_z * b_c[i]).real();
        a[i] = a_c[i].real();
    }
    // Normalize so a[0] == 1.
    double a0 = a[0];
    for (auto& v : b) v /= a0;
    for (auto& v : a) v /= a0;
    return {b, a};
}


// Returns {b, a} for a band-pass Butterworth filter of order N (per side),
// matching scipy.signal.butter(N, [Wn_low, Wn_high], btype="bandpass").
// The resulting digital filter has order 2*N (b, a each of length 2*N+1).
inline std::pair<std::vector<double>, std::vector<double>> design_bandpass(int N, double Wn_low, double Wn_high) {
    // Analog Butterworth prototype poles (cutoff = 1 rad/s, gain k=1), no zeros.
    std::vector<Cplx> p(N);
    for (int k = 0; k < N; k++) {
        double theta = kPi * (2.0 * k + N + 1) / (2.0 * N);
        p[k] = Cplx(std::cos(theta), std::sin(theta));
    }
    std::vector<Cplx> z; // lowpass prototype has no zeros
    double k_gain = 1.0;

    double fs = 2.0; // same convention as design_lowpass
    double warped_low  = 2.0 * fs * std::tan(kPi * Wn_low  / fs);
    double warped_high = 2.0 * fs * std::tan(kPi * Wn_high / fs);
    double bw = warped_high - warped_low;
    double wo = std::sqrt(warped_low * warped_high);

    // --- lp2bp (analog low-pass -> band-pass transform) ---
    int degree = (int)p.size() - (int)z.size(); // = N (extra zeros at origin)

    std::vector<Cplx> p_lp(p.size());
    for (size_t i = 0; i < p.size(); i++) p_lp[i] = p[i] * (bw / 2.0);

    std::vector<Cplx> p_bp, z_bp;
    for (auto& pk : p_lp) {
        Cplx term = std::sqrt(pk * pk - Cplx(wo * wo, 0.0));
        p_bp.push_back(pk + term);
    }
    for (auto& pk : p_lp) {
        Cplx term = std::sqrt(pk * pk - Cplx(wo * wo, 0.0));
        p_bp.push_back(pk - term);
    }
    for (int i = 0; i < degree; i++) z_bp.push_back(Cplx(0.0, 0.0)); // zeros at s=0

    double k_bp = k_gain * std::pow(bw, degree);

    // --- Bilinear transform to the z-plane ---
    double fs2 = 2.0 * fs;

    std::vector<Cplx> p_z(p_bp.size());
    Cplx denom_prod(1.0, 0.0);
    for (size_t i = 0; i < p_bp.size(); i++) {
        p_z[i] = (fs2 + p_bp[i]) / (fs2 - p_bp[i]);
        denom_prod *= (fs2 - p_bp[i]);
    }

    std::vector<Cplx> z_z(z_bp.size());
    Cplx numer_prod(1.0, 0.0);
    for (size_t i = 0; i < z_bp.size(); i++) {
        z_z[i] = (fs2 + z_bp[i]) / (fs2 - z_bp[i]);
        numer_prod *= (fs2 - z_bp[i]);
    }
    // Degree-difference zeros land at z = -1 (Nyquist).
    int extra_zeros = (int)p_bp.size() - (int)z_bp.size();
    for (int i = 0; i < extra_zeros; i++) z_z.push_back(Cplx(-1.0, 0.0));

    Cplx k_z = k_bp * (numer_prod / denom_prod);

    std::vector<Cplx> b_c = poly_from_roots(z_z);
    std::vector<Cplx> a_c = poly_from_roots(p_z);

    int order = (int)p_z.size(); // = 2*N
    std::vector<double> b(order + 1), a(order + 1);
    for (int i = 0; i <= order; i++) {
        b[i] = (k_z * b_c[i]).real();
        a[i] = a_c[i].real();
    }
    double a0 = a[0];
    for (auto& v : b) v /= a0;
    for (auto& v : a) v /= a0;
    return {b, a};
}

// Direct form II transposed IIR filter (matches scipy.signal.lfilter).
inline std::vector<double> lfilter(const std::vector<double>& b, const std::vector<double>& a,
                                    const std::vector<double>& x, std::vector<double> z) {
    size_t n_coef = std::max(b.size(), a.size());
    std::vector<double> bb(n_coef, 0.0), aa(n_coef, 0.0);
    for (size_t i = 0; i < b.size(); i++) bb[i] = b[i];
    for (size_t i = 0; i < a.size(); i++) aa[i] = a[i];
    if (z.size() != n_coef - 1) z.assign(n_coef - 1, 0.0);

    std::vector<double> y(x.size());
    for (size_t n = 0; n < x.size(); n++) {
        double xn = x[n];
        double yn = bb[0] * xn + (n_coef > 1 ? z[0] : 0.0);
        y[n] = yn;
        for (size_t i = 1; i < n_coef - 1; i++) {
            z[i - 1] = bb[i] * xn + z[i] - aa[i] * yn;
        }
        if (n_coef > 1) z[n_coef - 2] = bb[n_coef - 1] * xn - aa[n_coef - 1] * yn;
    }
    return y;
}

// Steady-state initial conditions for a step input (matches scipy.signal.lfilter_zi).
inline std::vector<double> lfilter_zi(const std::vector<double>& b, const std::vector<double>& a) {
    size_t n = std::max(a.size(), b.size());
    std::vector<double> aa(n, 0.0), bb(n, 0.0);
    for (size_t i = 0; i < a.size(); i++) aa[i] = a[i];
    for (size_t i = 0; i < b.size(); i++) bb[i] = b[i];

    int m = (int)n - 1;
    if (m <= 0) return {};
    Eigen::MatrixXd companionT = Eigen::MatrixXd::Zero(m, m);
    // companion(a).T: first row is 1's shifted (sub-diagonal in companion, so
    // its transpose is the superdiagonal), first column is -a[1:].
    for (int i = 0; i < m; i++) companionT(0, i) = -aa[i + 1];
    for (int i = 1; i < m; i++) companionT(i, i - 1) = 1.0;
    Eigen::MatrixXd IminusA = Eigen::MatrixXd::Identity(m, m) - companionT.transpose();

    Eigen::VectorXd B(m);
    for (int i = 0; i < m; i++) B(i) = bb[i + 1] - aa[i + 1] * bb[0];

    Eigen::VectorXd zi = IminusA.colPivHouseholderQr().solve(B);
    return std::vector<double>(zi.data(), zi.data() + zi.size());
}

// Zero-phase filtering matching scipy.signal.filtfilt(b, a, x) defaults
// (method="pad", padtype="odd", padlen=3*max(len(a),len(b))).
inline std::vector<double> filtfilt(const std::vector<double>& b, const std::vector<double>& a,
                                     const std::vector<double>& x) {
    int ntaps = (int)std::max(a.size(), b.size());
    int padlen = 3 * ntaps;
    int nx = (int)x.size();
    if (nx <= padlen) {
        throw std::runtime_error("filtfilt: signal too short for the default padlen");
    }

    // Odd extension on both ends.
    std::vector<double> ext(nx + 2 * padlen);
    for (int i = 0; i < padlen; i++) ext[i] = 2.0 * x[0] - x[padlen - i];
    for (int i = 0; i < nx; i++) ext[padlen + i] = x[i];
    for (int i = 0; i < padlen; i++) ext[padlen + nx + i] = 2.0 * x[nx - 1] - x[nx - 2 - i];

    std::vector<double> zi = lfilter_zi(b, a);

    std::vector<double> zi0 = zi;
    for (auto& v : zi0) v *= ext[0];
    std::vector<double> y1 = lfilter(b, a, ext, zi0);

    std::vector<double> y1r(y1.rbegin(), y1.rend());
    std::vector<double> zi1 = zi;
    for (auto& v : zi1) v *= y1r[0];
    std::vector<double> y2 = lfilter(b, a, y1r, zi1);
    std::vector<double> y2r(y2.rbegin(), y2.rend());

    return std::vector<double>(y2r.begin() + padlen, y2r.begin() + padlen + nx);
}

} // namespace butterworth
