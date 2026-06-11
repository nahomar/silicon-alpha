// heston_mc.cpp — fast Heston Monte-Carlo European-call pricer.
//
// Why this exists: the Heston MC reference is the bottleneck of the Phase-0
// validation harness (odte/eval/validate_dml.py spends most of its wall-clock
// here). This is a drop-in-faster C++ engine that (a) replicates the exact
// Euler full-truncation scheme of odte/synth_options.py so results agree with
// the Python reference within Monte-Carlo error, (b) adds antithetic variates
// for variance reduction the Python path lacks, and (c) self-validates against
// the closed-form Black-Scholes limit (xi -> 0, v0 = sigma^2).
//
// Scheme (matches odte/synth_options.py::Heston.simulate_paths):
//   z1, z2 ~ N(0,1) iid
//   w_v   = rho*z1 + sqrt(1-rho^2)*z2
//   v'    = max( v + kappa*(theta - max(v,0))*dt + xi*sqrt(max(v,0))*sqrt_dt*w_v , 0 )
//   S'    = S * exp( (r - 0.5 v) dt + sqrt(v)*sqrt_dt*z1 )
//   price = exp(-r T) * mean( max(S_T - K, 0) )
//
// Build:  make            (auto-detects Homebrew libomp; serial otherwise)
// Run:    ./heston_mc --selftest
//         ./heston_mc --benchmark
//         ./heston_mc --price S K T sigma [paths steps kappa theta xi rho r]
//
// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ---- xoshiro256++ PRNG (fast, high quality) + splitmix64 seeding ----------
struct Rng {
    uint64_t s[4];
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    explicit Rng(uint64_t seed) {
        // splitmix64 to fill state
        for (int i = 0; i < 4; ++i) {
            uint64_t z = (seed += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }
    inline uint64_t next() {
        const uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    // uniform double in (0,1)
    inline double uniform() {
        return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
};

// Box-Muller: returns a pair of independent standard normals.
struct NormalPair { double a, b; };
inline NormalPair box_muller(Rng& rng) {
    double u1 = rng.uniform(), u2 = rng.uniform();
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 6.283185307179586 * u2;
    return {r * std::cos(theta), r * std::sin(theta)};
}

struct Heston {
    double kappa = 4.0, theta = 0.04, xi = 0.6, rho = -0.7, v0 = 0.04, r = 0.0;
};

// Evolve one path given a draw callback that yields (z1, z2) per step.
// `sign` lets the caller flip the entire normal sequence for the antithetic
// twin without re-drawing. Returns the discounted-at-caller terminal S.
template <class DrawFn>
inline double terminal_S(double S0, double T, int steps, const Heston& h,
                         DrawFn draw, double sign) {
    const double dt = T / steps;
    const double sqrt_dt = std::sqrt(dt);
    const double rho2 = std::sqrt(1.0 - h.rho * h.rho);
    double S = S0, v = h.v0;
    for (int t = 0; t < steps; ++t) {
        NormalPair zp = draw();
        double z1 = sign * zp.a, z2 = sign * zp.b;
        double w_v = h.rho * z1 + rho2 * z2;
        double v_pos = v > 0.0 ? v : 0.0;
        double sqrt_v = std::sqrt(v_pos);
        double v_next = v + h.kappa * (h.theta - v_pos) * dt + h.xi * sqrt_v * sqrt_dt * w_v;
        if (v_next < 0.0) v_next = 0.0;
        S = S * std::exp((h.r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1);
        v = v_next;
    }
    return S;
}

struct PriceResult { double price, stderr_, paths_per_sec; long n_samples; };

// Monte-Carlo call price. With antithetic=true the sampling unit is a
// (path, mirrored-path) pair, which both reduces variance and makes the
// reported standard error reflect that reduction.
PriceResult heston_call(double S0, double K, double T, const Heston& h,
                        long n_paths, int n_steps, uint64_t seed,
                        bool antithetic) {
    const double disc = std::exp(-h.r * T);
    long n_units = antithetic ? n_paths / 2 : n_paths;
    if (n_units < 1) n_units = 1;

    double sum = 0.0, sumsq = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        double local_sum = 0.0, local_sumsq = 0.0;
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
#else
        int tid = 0;
#endif
        for (long i = 0; i < n_units; ++i) {
            Rng rng(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1))
                    ^ (0xD1B54A32D192ED03ULL * (uint64_t)(tid + 1)));
            double sample;
            if (antithetic) {
                // Draw a per-step normal buffer once, reuse negated for the twin.
                double buf_a[256], buf_b[256];
                int steps = n_steps <= 256 ? n_steps : 256;
                for (int t = 0; t < steps; ++t) {
                    NormalPair zp = box_muller(rng);
                    buf_a[t] = zp.a; buf_b[t] = zp.b;
                }
                int idx = 0;
                auto draw_fwd = [&]() -> NormalPair { NormalPair p{buf_a[idx], buf_b[idx]}; ++idx; return p; };
                idx = 0;
                double ST1 = terminal_S(S0, T, steps, h, draw_fwd, +1.0);
                idx = 0;
                double ST2 = terminal_S(S0, T, steps, h, draw_fwd, -1.0);
                double p1 = ST1 > K ? ST1 - K : 0.0;
                double p2 = ST2 > K ? ST2 - K : 0.0;
                sample = disc * 0.5 * (p1 + p2);
            } else {
                auto draw = [&]() -> NormalPair { return box_muller(rng); };
                double ST = terminal_S(S0, T, n_steps, h, draw, +1.0);
                double p = ST > K ? ST - K : 0.0;
                sample = disc * p;
            }
            local_sum += sample;
            local_sumsq += sample * sample;
        }
#ifdef _OPENMP
        #pragma omp critical
#endif
        { sum += local_sum; sumsq += local_sumsq; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double mean = sum / (double)n_units;
    double var = sumsq / (double)n_units - mean * mean;
    if (var < 0) var = 0;
    double stderr_ = std::sqrt(var / (double)n_units);
    long total_paths = antithetic ? n_units * 2 : n_units;
    return {mean, stderr_, secs > 0 ? total_paths / secs : 0.0, n_units};
}

// Analytic Black-Scholes call (for the xi -> 0 self-test).
double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.7071067811865476); }
double bs_call(double S, double K, double T, double sigma, double r) {
    if (sigma <= 0 || T <= 0) return std::max(S - K, 0.0);
    double srt = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / srt;
    double d2 = d1 - srt;
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

int self_test() {
    // In the xi -> 0 limit with v0 = sigma^2, Heston collapses to Black-Scholes
    // at constant vol sigma. The MC price must agree with the closed form to
    // within a few standard errors. This is a correctness gate against a known
    // analytic answer, not a parity-with-numpy hand-wave.
    printf("[selftest] Heston(xi=0) vs analytic Black-Scholes\n");
    struct Case { double S, K, T, sigma; };
    Case cases[] = {{5500, 5500, 30.0/365, 0.20}, {5500, 5600, 10.0/365, 0.20},
                    {5500, 5400, 5.0/365, 0.35}, {5500, 5500, 60.0/365, 0.10}};
    int fails = 0;
    for (auto c : cases) {
        Heston h; h.xi = 0.0; h.rho = 0.0; h.v0 = c.sigma * c.sigma;
        h.kappa = 0.0; h.theta = c.sigma * c.sigma; h.r = 0.0;
        auto res = heston_call(c.S, c.K, c.T, h, 4'000'000, 64, 12345, true);
        double bs = bs_call(c.S, c.K, c.T, c.sigma, 0.0);
        double z = std::fabs(res.price - bs) / (res.stderr_ + 1e-12);
        bool ok = z < 5.0;  // within 5 sigma
        printf("  S=%.0f K=%.0f T=%.4f sig=%.2f  MC=%.4f  BS=%.4f  |z|=%.2f  %s\n",
               c.S, c.K, c.T, c.sigma, res.price, bs, z, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    }
    // Antithetic variance-reduction check at equal path budget.
    Heston h;  // default Heston (xi=0.6, rho=-0.7)
    long N = 2'000'000;
    auto plain = heston_call(5500, 5500, 30.0/365, h, N, 64, 999, false);
    auto anti  = heston_call(5500, 5500, 30.0/365, h, N, 64, 999, true);
    double vr = (plain.stderr_ * plain.stderr_) / (anti.stderr_ * anti.stderr_ + 1e-30);
    printf("[selftest] antithetic variance reduction @ %ld paths: %.2fx "
           "(plain se=%.4f, anti se=%.4f)\n", N, vr, plain.stderr_, anti.stderr_);
    bool vr_ok = vr > 1.05;  // must actually reduce variance
    if (!vr_ok) { printf("  FAIL: antithetic did not reduce variance\n"); ++fails; }

    printf(fails == 0 ? "[selftest] PASS\n" : "[selftest] FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}

int benchmark() {
    Heston h;
    const long N = 4'000'000; const int steps = 64;
    printf("[bench] Heston call, %ld paths x %d steps", N, steps);
#ifdef _OPENMP
    printf("  (OpenMP, %d threads)\n", omp_get_max_threads());
#else
    printf("  (serial)\n");
#endif
    auto res = heston_call(5500, 5500, 30.0/365, h, N, steps, 42, true);
    printf("  price=%.4f  stderr=%.5f  throughput=%.2f Mpaths/s  (%ld pair-samples)\n",
           res.price, res.stderr_, res.paths_per_sec / 1e6, res.n_samples);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0) return self_test();
    if (argc >= 2 && std::strcmp(argv[1], "--benchmark") == 0) return benchmark();
    if (argc >= 6 && std::strcmp(argv[1], "--price") == 0) {
        // --price S K T sigma [paths steps kappa theta xi rho r]
        double S = atof(argv[2]), K = atof(argv[3]), T = atof(argv[4]), sigma = atof(argv[5]);
        long paths = argc > 6 ? atol(argv[6]) : 1'000'000;
        int steps = argc > 7 ? atoi(argv[7]) : 48;
        Heston h; h.v0 = sigma * sigma; h.theta = sigma * sigma;
        if (argc > 8) h.kappa = atof(argv[8]);
        if (argc > 9) h.theta = atof(argv[9]);
        if (argc > 10) h.xi = atof(argv[10]);
        if (argc > 11) h.rho = atof(argv[11]);
        if (argc > 12) h.r = atof(argv[12]);
        auto res = heston_call(S, K, T, h, paths, steps, 7, true);
        // machine-readable single line: price stderr
        printf("%.8f %.8f\n", res.price, res.stderr_);
        return 0;
    }
    fprintf(stderr,
            "usage:\n  %s --selftest\n  %s --benchmark\n"
            "  %s --price S K T sigma [paths steps kappa theta xi rho r]\n",
            argv[0], argv[0], argv[0]);
    return 2;
}
