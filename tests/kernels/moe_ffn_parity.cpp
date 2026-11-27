// tests/kernels/moe_ffn_parity.cpp - the routed-expert kernels of moe_ffn.hpp at the Qwen3.8-35B-A3B geometry (hidden 2048, expert
// width 512, 256 experts, 8 used) on random weights and activations, against a CPU reference in double precision.
//
//   - top_k, router, combine, group: against their definitions on the CPU.
//   - quantize + gate_up and down: against the dequantized weights with unrounded activations (the error is the INT8
//     activation rounding, per 16 values), and against a CPU model that rounds the activations the same way (the error is the
//     float summation order).
//   - every kernel: a token's rows are bitwise the same in a window of 1, 8 and 32 rows and at any position in it.
//   - a microbench line per kernel: microseconds per call, and the weight bytes the pairs read per second.
#include "basalt/kernels/moe_ffn.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace moe = basalt::kernels::moe;

namespace {
constexpr int H = 2048, F = 512, E = 256, K = 8;
std::mt19937 rng(11);
int fails = 0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

template <class T> T* up(const std::vector<T>& h) {
    T* d;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> std::vector<T> down_(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

double ue4m3(unsigned c) {
    const unsigned e = c >> 3, m = c & 7;
    return e == 0 ? m / 512.0 : (8 + m) * std::ldexp(1.0, (int) e - 10);
}
const double kE2m1[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};

// a random NVFP4 matrix of rows x cols: codes, UE4M3 scales; bytes in the blob's layout (blocks of 64: four scales, then 32 code
// bytes where group s keeps value j in the low nibble of byte 8 s + j and value j + 8 in the high one)
struct Mat {
    int rows, cols;
    std::vector<uint8_t> bytes;
    double scale = 1.0;                                   // the tensor scale
    double at(int r, int c) const {
        const uint8_t* blk = &bytes[((size_t) r * (cols / 64) + c / 64) * 36];
        const int sub = (c % 64) / 16, j = c % 16;
        const uint8_t b = blk[4 + 8 * sub + (j % 8)];
        return kE2m1[j < 8 ? b & 15 : b >> 4] * ue4m3(blk[sub]) * scale;
    }
};
Mat random_mat(int rows, int cols, double scale) {
    Mat m{rows, cols, std::vector<uint8_t>((size_t) rows * (cols / 64) * 36), scale};
    for (size_t b = 0; b < m.bytes.size() / 36; ++b) {
        uint8_t* blk = &m.bytes[b * 36];
        for (int s = 0; s < 4; ++s) blk[s] = (uint8_t) (40 + rng() % 70);   // moderate scales, none the NaN code
        for (int i = 4; i < 36; ++i) blk[i] = (uint8_t) rng();
    }
    return m;
}

struct Stat {
    double max_abs = 0, max_rel = 0, sum_sq = 0, sum_ref = 0;
    void add(double got, double ref) {
        const double d = std::fabs(got - ref);
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
        sum_ref += ref * ref;
    }
    void finish(double ref_max) { max_rel = max_abs / std::max(ref_max, 1e-30); }
};

void report(const char* what, const char* vs, const Stat& s, double tol, double ref_max) {
    Stat t = s;
    t.finish(ref_max);
    const bool ok = t.max_rel <= tol;
    std::printf("  %-26s vs %-34s max abs %.3e  max rel to max|ref| %.3e  rms rel %.3e  %s\n", what, vs, t.max_abs, t.max_rel,
                std::sqrt(t.sum_sq / std::max(t.sum_ref, 1e-30)), ok ? "ok" : "FAIL");
    fails += ok ? 0 : 1;
}

float u2f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float bf16_to_f(uint16_t b) { return u2f((uint32_t) b << 16); }

// CPU INT8 rounding of a row as the quantize kernel does it: a scale amax / 127 per 16 values, codes rounded to nearest even
void quantize_ref(const std::vector<float>& x, std::vector<float>& xr) {
    xr.resize(x.size());
    for (size_t g = 0; g < x.size() / 16; ++g) {
        float amax = 0;
        for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[g * 16 + i]));
        const float sc = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
        for (int i = 0; i < 16; ++i) xr[g * 16 + i] = (float) std::nearbyint(x[g * 16 + i] * inv) * sc;
    }
}
}  // namespace

int main() {
    const size_t blob = moe::blob_bytes(H, F), tail_at = moe::tail_offset(H, F);
    const size_t mat_bytes = (size_t) F * (H / 64) * 36;                  // gate (and up, and down: the same size)
    std::printf("qwen35moe expert kernels: hidden %d, expert width %d, %d experts, %d used; blob %zu bytes\n", H, F, E, K, blob);

    // ---- the experts: random matrices in the blob layout
    std::vector<uint8_t> hexperts((size_t) E * blob, 0);
    std::vector<Mat> gate, upm, dn;
    std::vector<double> sg(E), su(E), sd(E);
    for (int e = 0; e < E; ++e) {
        gate.push_back(random_mat(F, H, 0.0)); upm.push_back(random_mat(F, H, 0.0)); dn.push_back(random_mat(H, F, 0.0));
        sg[e] = 1e-3 * (0.5 + (rng() % 1000) / 1000.0);
        su[e] = 1e-3 * (0.5 + (rng() % 1000) / 1000.0);
        sd[e] = 1e-3 * (0.5 + (rng() % 1000) / 1000.0);
        gate[e].scale = sg[e]; upm[e].scale = su[e]; dn[e].scale = sd[e];
        uint8_t* b = &hexperts[(size_t) e * blob];
        std::memcpy(b, gate[e].bytes.data(), mat_bytes);
        std::memcpy(b + mat_bytes, upm[e].bytes.data(), mat_bytes);
        std::memcpy(b + 2 * mat_bytes, dn[e].bytes.data(), mat_bytes);
        const float tail[4] = {(float) sg[e], (float) su[e], (float) sd[e], 0.0f};
        std::memcpy(b + tail_at, tail, 16);
        sg[e] = tail[0]; su[e] = tail[1]; sd[e] = tail[2];
        gate[e].scale = sg[e]; upm[e].scale = su[e]; dn[e].scale = sd[e];
    }
    uint8_t* dexp = up(hexperts);
    const moe::Experts ex{dexp, blob, E, H, F};

    const int T = 12, P = T * K;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) T * H);
    for (float& v : x) v = nd(rng);
    for (int t = 0; t < T; ++t)                                            // a few outliers, as real activations have
        x[(size_t) t * H + rng() % H] *= 12.0f;
    std::vector<int32_t> idx((size_t) P);
    for (int t = 0; t < T; ++t) {
        std::vector<int> all(E);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        for (int k = 0; k < K; ++k) idx[(size_t) t * K + k] = all[(size_t) k];
    }
    float* dx = up(x);
    int32_t* didx = up(idx);

    // ---- gate_up: x -> INT8 -> silu(gate) * up
    std::printf("gate_up (silu(gate x) * (up x)), %d pairs:\n", P);
    int8_t* dxq; float* dxs;
    CK(cudaMalloc(&dxq, (size_t) T * H));
    CK(cudaMalloc(&dxs, (size_t) T * H / 16 * 4));
    moe::quantize(dx, T, H, dxq, dxs, nullptr);
    float* dh;
    CK(cudaMalloc(&dh, (size_t) P * F * 4));
    moe::gate_up(ex, dxq, dxs, didx, T, K, dh, nullptr);
    CK(cudaDeviceSynchronize());
    const auto h = down_(dh, (size_t) P * F);
    Stat s_exact, s_int8;
    double ref_max = 0;
    std::vector<std::vector<double>> href((size_t) P, std::vector<double>(F));
    for (int p = 0; p < P; p += 3) {                                       // a third of the pairs (the CPU is double precision)
        const int t = p / K, e = idx[(size_t) p];
        std::vector<float> xt(x.begin() + (size_t) t * H, x.begin() + (size_t) (t + 1) * H), xr;
        quantize_ref(xt, xr);
        for (int r = 0; r < F; ++r) {
            double g = 0, u = 0, gq = 0, uq = 0;
            for (int c = 0; c < H; ++c) {
                const double wg = gate[e].at(r, c), wu = upm[e].at(r, c);
                g += wg * xt[c]; u += wu * xt[c];
                gq += wg * xr[c]; uq += wu * xr[c];
            }
            const double ref = g / (1 + std::exp(-g)) * u, refq = gq / (1 + std::exp(-gq)) * uq;
            ref_max = std::max(ref_max, std::fabs(ref));
            s_exact.add(h[(size_t) p * F + r], ref);
            s_int8.add(h[(size_t) p * F + r], refq);
        }
    }
    report("gate_up", "double, unrounded activations", s_exact, 0.05, ref_max);
    report("gate_up", "double, same INT8 activation rounding", s_int8, 1e-4, ref_max);

    // ---- down
    std::printf("down (down x h), %d pairs:\n", P);
    float* dy;
    CK(cudaMalloc(&dy, (size_t) P * H * 4));
    moe::down(ex, dh, didx, T, K, dy, nullptr);
    CK(cudaDeviceSynchronize());
    const auto y = down_(dy, (size_t) P * H);
    Stat d_exact, d_int8;
    double dref_max = 0;
    for (int p = 0; p < P; p += 3) {
        const int e = idx[(size_t) p];
        std::vector<float> hp(h.begin() + (size_t) p * F, h.begin() + (size_t) (p + 1) * F), hr;
        quantize_ref(hp, hr);
        for (int c = 0; c < H; c += 3) {
            double a = 0, b = 0;
            for (int r = 0; r < F; ++r) {
                const double w = dn[e].at(c, r);
                a += w * hp[r];
                b += w * hr[r];
            }
            dref_max = std::max(dref_max, std::fabs(a));
            d_exact.add(y[(size_t) p * H + c], a);
            d_int8.add(y[(size_t) p * H + c], b);
        }
    }
    report("down", "double, unrounded activations", d_exact, 0.05, dref_max);
    report("down", "double, same INT8 activation rounding", d_int8, 1e-4, dref_max);

    // ---- router and top_k
    std::printf("router (BF16 rows . x), top_k (softmax, 8 largest, renormalised):\n");
    const int RR = E + 1;
    std::vector<uint16_t> w((size_t) RR * H);
    for (uint16_t& v : w) v = (uint16_t) (f2u(0.04f * nd(rng)) >> 16);
    uint16_t* dw = up(w);
    float* dlog;
    CK(cudaMalloc(&dlog, (size_t) T * RR * 4));
    moe::router(dx, dw, T, H, RR, dlog, RR, nullptr);
    CK(cudaDeviceSynchronize());
    const auto logits = down_(dlog, (size_t) T * RR);
    Stat r_stat;
    double lmax = 0;
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < RR; ++r) {
            double a = 0;
            for (int c = 0; c < H; ++c) a += (double) bf16_to_f(w[(size_t) r * H + c]) * x[(size_t) t * H + c];
            lmax = std::max(lmax, std::fabs(a));
            r_stat.add(logits[(size_t) t * RR + r], a);
        }
    report("router", "double dot", r_stat, 1e-5, lmax);
    int32_t *dtk; float* dtw;
    CK(cudaMalloc(&dtk, (size_t) T * K * 4));
    CK(cudaMalloc(&dtw, (size_t) T * K * 4));
    moe::top_k(dlog, RR, T, E, K, dtk, dtw, nullptr);
    CK(cudaDeviceSynchronize());
    const auto tk = down_(dtk, (size_t) T * K);
    const auto tw = down_(dtw, (size_t) T * K);
    Stat k_stat;
    int wrong_idx = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> p(E);
        double mx = -1e30, z = 0;
        for (int e = 0; e < E; ++e) mx = std::max(mx, (double) logits[(size_t) t * RR + e]);
        for (int e = 0; e < E; ++e) z += (p[e] = std::exp(logits[(size_t) t * RR + e] - mx));
        for (double& v : p) v /= z;
        std::vector<int> ord(E);
        std::iota(ord.begin(), ord.end(), 0);
        std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) { return p[a] > p[b]; });
        double sum = 0;
        for (int k = 0; k < K; ++k) sum += p[ord[k]];
        sum = std::max(sum, 6.103515625e-5);
        for (int k = 0; k < K; ++k) {
            wrong_idx += tk[(size_t) t * K + k] != ord[k];
            k_stat.add(tw[(size_t) t * K + k], p[ord[k]] / sum);
        }
    }
    std::printf("  top_k experts: %d of %d differ from the CPU's\n", wrong_idx, T * K);
    fails += wrong_idx != 0;
    report("top_k weights", "double softmax", k_stat, 1e-5, 1.0);

    // ---- combine
    std::printf("combine (BF16 residual + routed + sigmoid(gate) shared):\n");
    std::vector<uint16_t> res((size_t) T * H);
    for (uint16_t& v : res) v = (uint16_t) (f2u(nd(rng)) >> 16);
    std::vector<float> shared((size_t) T * H);
    for (float& v : shared) v = 0.5f * nd(rng);
    uint16_t* dres = up(res);
    float* dsh = up(shared);
    moe::combine(dres, dy, dtw, dlog + E, RR, dsh, T, K, H, nullptr);
    CK(cudaDeviceSynchronize());
    const auto out = down_(dres, (size_t) T * H);
    Stat c_stat;
    int ulp_off = 0;
    double cmax = 0;
    for (int t = 0; t < T; ++t) {
        const double sig = 1.0 / (1.0 + std::exp(-(double) logits[(size_t) t * RR + E]));
        for (int c = 0; c < H; ++c) {
            double routed = 0;
            for (int k = 0; k < K; ++k) routed += (double) tw[(size_t) t * K + k] * y[((size_t) t * K + k) * H + c];
            const double ref = (double) bf16_to_f(res[(size_t) t * H + c]) + routed + sig * shared[(size_t) t * H + c];
            const double got = bf16_to_f(out[(size_t) t * H + c]);
            c_stat.add(got, ref);
            cmax = std::max(cmax, std::fabs(ref));
            ulp_off += std::fabs(got - ref) > std::fabs(ref) * 0.00392 + 1e-5;    // half a BF16 step is at most 2^-8 of the value; 1e-5 is the float32 sum's error where terms cancel
        }
    }
    std::printf("  values more than half a BF16 step from the double reference: %d of %d\n", ulp_off, T * H);
    fails += ulp_off != 0;
    report("combine", "double, rounded once to BF16", c_stat, 2.0 / 256, cmax);

    // ---- group
    std::printf("group (pairs by expert):\n");
    int32_t *db, *dt, *dd, *de;
    CK(cudaMalloc(&db, (E + 1) * 4)); CK(cudaMalloc(&dt, P * 4)); CK(cudaMalloc(&dd, P * 4)); CK(cudaMalloc(&de, P * 4));
    moe::group(didx, T, K, E, db, dt, dd, de, nullptr);
    CK(cudaDeviceSynchronize());
    const auto bounds = down_(db, E + 1), gt = down_(dt, P), gd = down_(dd, P), ge = down_(de, P);
    std::vector<int> cnt(E, 0), seen(P, 0);
    for (int p = 0; p < P; ++p) ++cnt[(size_t) idx[(size_t) p]];
    int bad = bounds[E] != P;
    for (int e = 0; e < E; ++e) bad += bounds[e + 1] - bounds[e] != cnt[e];
    for (int i = 0; i < P; ++i) {
        bad += gt[i] != gd[i] / K || ge[i] != idx[(size_t) gd[i]];
        ++seen[(size_t) gd[i]];
    }
    for (int p = 0; p < P; ++p) bad += seen[p] != 1;
    std::printf("  runs, token rows, pairs and experts that disagree with the CPU: %d  %s\n", bad, bad ? "FAIL" : "ok");
    fails += bad != 0;

    // ---- a token's rows do not depend on the window's width or its place in it
    std::printf("window-width invariance (token 5 alone, in 8 rows at row 3, in 12 rows at row 5; bitwise):\n");
    auto solo_rows = [&](int t0, int n, float* h_out, float* y_out, float* lg_out, int32_t* tk_out, float* tw_out) {
        int8_t* q; float* s; float *h2, *y2, *l2, *w2; int32_t *i2;
        CK(cudaMalloc(&q, (size_t) n * H)); CK(cudaMalloc(&s, (size_t) n * H / 16 * 4));
        CK(cudaMalloc(&h2, (size_t) n * K * F * 4)); CK(cudaMalloc(&y2, (size_t) n * K * H * 4));
        CK(cudaMalloc(&l2, (size_t) n * RR * 4)); CK(cudaMalloc(&w2, (size_t) n * K * 4)); CK(cudaMalloc(&i2, (size_t) n * K * 4));
        moe::router(dx + (size_t) t0 * H, dw, n, H, RR, l2, RR, nullptr);
        moe::top_k(l2, RR, n, E, K, i2, w2, nullptr);
        moe::quantize(dx + (size_t) t0 * H, n, H, q, s, nullptr);
        moe::gate_up(ex, q, s, didx + (size_t) t0 * K, n, K, h2, nullptr);
        moe::down(ex, h2, didx + (size_t) t0 * K, n, K, y2, nullptr);
        CK(cudaDeviceSynchronize());
        std::memcpy(h_out, down_(h2, (size_t) n * K * F).data(), (size_t) n * K * F * 4);
        std::memcpy(y_out, down_(y2, (size_t) n * K * H).data(), (size_t) n * K * H * 4);
        std::memcpy(lg_out, down_(l2, (size_t) n * RR).data(), (size_t) n * RR * 4);
        std::memcpy(tk_out, down_(i2, (size_t) n * K).data(), (size_t) n * K * 4);
        std::memcpy(tw_out, down_(w2, (size_t) n * K).data(), (size_t) n * K * 4);
        cudaFree(q); cudaFree(s); cudaFree(h2); cudaFree(y2); cudaFree(l2); cudaFree(w2); cudaFree(i2);
    };
    // a window of rows t0 .. t0 + n holding token 5 at row 5 - t0; the same token's outputs must not change
    std::vector<float> h1((size_t) K * F), y1((size_t) K * H), l1(RR), w1(K);
    std::vector<int32_t> k1(K);
    solo_rows(5, 1, h1.data(), y1.data(), l1.data(), k1.data(), w1.data());
    int variants_bad = 0;
    for (const auto [t0, n] : {std::pair<int, int>{2, 8}, {0, 12}, {5, 7}}) {
        std::vector<float> hh((size_t) n * K * F), yy((size_t) n * K * H), ll((size_t) n * RR), ww((size_t) n * K);
        std::vector<int32_t> kk((size_t) n * K);
        solo_rows(t0, n, hh.data(), yy.data(), ll.data(), kk.data(), ww.data());
        const int row = 5 - t0;
        const bool same = !std::memcmp(&hh[(size_t) row * K * F], h1.data(), K * F * 4) && !std::memcmp(&yy[(size_t) row * K * H], y1.data(), K * H * 4) &&
                          !std::memcmp(&ll[(size_t) row * RR], l1.data(), RR * 4) && !std::memcmp(&kk[(size_t) row * K], k1.data(), K * 4) &&
                          !std::memcmp(&ww[(size_t) row * K], w1.data(), K * 4);
        std::printf("  window of %2d rows, token at row %d: router, top_k, gate_up and down rows %s\n", n, row, same ? "bitwise equal" : "DIFFER");
        variants_bad += !same;
    }
    fails += variants_bad != 0;

    // ---- microbench: microseconds a call and the weight bytes the pairs read a second
    std::printf("microbench (RTX power limit as set on the card), per call:\n");
    auto bench = [&](const char* name, int rows, auto&& fn, double bytes) {
        for (int i = 0; i < 5; ++i) fn();
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        const int it = 200;
        for (int i = 0; i < it; ++i) fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        const double us = 1000.0 * ms / it;
        std::printf("  %-10s %2d rows  %8.1f us   %s%7.1f GB/s of expert weights read\n", name, rows, us, bytes > 0 ? "" : "(not weight bound) ", bytes > 0 ? bytes / us * 1e-3 : 0.0);
    };
    for (int rows : {1, 8, 24, 32}) {
        // pairs spread over distinct experts the way a real window's are: rows * K pairs
        std::vector<int32_t> bi((size_t) rows * K);
        for (size_t i = 0; i < bi.size(); ++i) bi[i] = (int32_t) (rng() % E);
        int32_t* dbi = up(bi);
        float *bx, *bh, *by; int8_t* bq; float* bs;
        CK(cudaMalloc(&bx, (size_t) rows * H * 4)); CK(cudaMemset(bx, 0, (size_t) rows * H * 4));
        CK(cudaMalloc(&bq, (size_t) rows * H)); CK(cudaMalloc(&bs, (size_t) rows * H / 16 * 4));
        CK(cudaMalloc(&bh, (size_t) rows * K * F * 4)); CK(cudaMalloc(&by, (size_t) rows * K * H * 4));
        moe::quantize(dx, rows, H, bq, bs, nullptr);
        std::vector<int> d(bi.begin(), bi.end());
        std::sort(d.begin(), d.end());
        const double distinct = (double) (std::unique(d.begin(), d.end()) - d.begin());
        bench("gate_up", rows, [&] { moe::gate_up(ex, bq, bs, dbi, rows, K, bh, nullptr); }, distinct * 2 * mat_bytes);
        bench("down", rows, [&] { moe::down(ex, bh, dbi, rows, K, by, nullptr); }, distinct * mat_bytes);
        cudaFree(dbi); cudaFree(bx); cudaFree(bq); cudaFree(bs); cudaFree(bh); cudaFree(by);
    }
    for (int rows : {1, 8, 32}) {
        float* l2; int32_t* i2; float* w2;
        CK(cudaMalloc(&l2, (size_t) rows * RR * 4)); CK(cudaMemset(l2, 0, (size_t) rows * RR * 4));
        CK(cudaMalloc(&i2, (size_t) rows * K * 4)); CK(cudaMalloc(&w2, (size_t) rows * K * 4));
        bench("router", rows, [&] { moe::router(dx, dw, rows, H, RR, l2, RR, nullptr); }, 0);
        bench("top_k", rows, [&] { moe::top_k(l2, RR, rows, E, K, i2, w2, nullptr); }, 0);
        cudaFree(l2); cudaFree(i2); cudaFree(w2);
    }
    std::printf("%s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
