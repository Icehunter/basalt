// tests/models/qwen35/moe_parity_test.cpp - a qwen35moe pack's routed experts: the per-pair INT8 kernels (decode) against the
// grouped products (prompt chunks: the FP4 block-scaled path the engine runs, and the MMQ path it replaced), on the first
// layer's real experts with the same routing, and a double-precision host reference.  Usage:
//   qwen35_moe_parity_test PACK.basalt [rows]     (skipped, exit 77, without a pack)
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/moe_ffn.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace moe = basalt::kernels::moe;
namespace mmq = basalt::prefill::mmq;
namespace fp4 = basalt::prefill::fp4;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) return 77;
    const int H = 2048, F = 512, E = 256, K = 8, N = argc > 2 ? std::atoi(argv[2]) : 256, kType = 40;
    basalt::core::ModelFile file;
    basalt::core::ModelSources srcs;
    std::string err;
    if (!file.open(argv[1], err)) { std::printf("open: %s\n", err.c_str()); return 77; }
    basalt::core::model_sources(file, srcs);
    const size_t blob = moe::blob_bytes(H, F), layer = blob * E;
    std::vector<uint8_t> host(layer);
    {
        std::ifstream f(argv[1], std::ios::binary);
        f.seekg((std::streamoff) srcs.experts.offset);
        f.read((char*) host.data(), (std::streamsize) layer);
        if (!f) { std::printf("short read of the experts\n"); return 1; }
    }
    for (int e : {0, 62})
        for (int r : {0, 1, 2, 3}) {
            const uint8_t* row = host.data() + (size_t) e * blob + (size_t) r * 1152;
            int nz = 0;
            for (int i = 0; i < 1152; ++i) nz += row[i] != 0;
            std::printf("host expert %2d row %d: %4d of 1152 bytes nonzero, first bytes %02x %02x %02x %02x %02x %02x %02x %02x\n", e, r, nz, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);
        }
    uint8_t* dev = nullptr;
    CK(cudaMalloc(&dev, layer));
    CK(cudaMemcpy(dev, host.data(), layer, cudaMemcpyHostToDevice));
    const moe::Experts ex{dev, blob, E, H, F};

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) N * H), w((size_t) N * K);
    std::vector<int32_t> idx((size_t) N * K);
    for (float& v : x) v = nd(rng);
    for (int t = 0; t < N; ++t) {
        std::vector<int> all(E);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        for (int k = 0; k < K; ++k) idx[(size_t) t * K + k] = all[(size_t) k];
    }
    const int P = N * K;
    float *dx, *dh, *dy_old, *dy_new, *dgu, *dxs;
    int8_t* dxq;
    int32_t *didx, *dbounds, *dtok, *ddst, *dexp, *dident;
    void* dmx;
    CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dxq, (size_t) N * H));
    CK(cudaMalloc(&dxs, (size_t) N * H / 16 * 4));
    CK(cudaMalloc(&dh, (size_t) P * F * 4));
    CK(cudaMalloc(&dy_old, (size_t) P * H * 4));
    CK(cudaMalloc(&dy_new, (size_t) P * H * 4));
    CK(cudaMalloc(&dgu, (size_t) P * 2 * F * 4));
    CK(cudaMalloc(&didx, idx.size() * 4));
    CK(cudaMalloc(&dbounds, (E + 1) * 4));
    CK(cudaMalloc(&dtok, (size_t) P * 4));
    CK(cudaMalloc(&ddst, (size_t) P * 4));
    CK(cudaMalloc(&dexp, (size_t) P * 4));
    CK(cudaMalloc(&dident, (size_t) P * 4));
    CK(cudaMalloc(&dmx, std::max(mmq::q8_bytes(P, H), mmq::q8_bytes(P, F))));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    mmq::iota(dident, P, nullptr);

    // the per-pair path
    moe::quantize(dx, N, H, dxq, dxs, nullptr);
    moe::gate_up(ex, dxq, dxs, didx, N, K, dh, nullptr);
    moe::down(ex, dh, didx, N, K, dy_old, nullptr);
    CK(cudaDeviceSynchronize());

    // the grouped path (as forward.cpp's moe_apply runs it)
    mmq::load(kType, 2 * F);
    mmq::load(kType, H);
    mmq::Context ctx;
    moe::group(didx, N, K, E, dbounds, dtok, ddst, dexp, nullptr);
    mmq::quantize(dx, dtok, dmx, kType, H, H, P, nullptr);
    mmq::Product p;
    p.w = dev;
    p.type = kType;
    p.w_rows = 2 * F;
    p.w_cols = H;
    p.expert_bytes = blob;
    p.n = E;
    p.xq = dmx;
    p.bounds = dbounds;
    p.ids = dident;
    p.total_rows = P;
    p.max_rows = N;
    p.opt_rows = (P + E - 1) / E;
    p.dst = dgu;
    p.ld_dst = 2 * F;
    ctx.run(p, nullptr);
    mmq::SwigluForm form;
    form.row_expert = dexp;
    form.tail_stride = blob;
    mmq::swiglu(dgu, dh, P, F, false, nullptr, (const float*) (dev + moe::tail_offset(H, F)), nullptr, 0, form);
    {   // the raw gate product (before the tail scales) against a host decode of the blob
        CK(cudaDeviceSynchronize());
        std::vector<float> gu((size_t) P * 2 * F);
        std::vector<int32_t> tok(P), exv(P);
        CK(cudaMemcpy(gu.data(), dgu, gu.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tok.data(), dtok, (size_t) P * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(exv.data(), dexp, (size_t) P * 4, cudaMemcpyDeviceToHost));
        auto ue = [](unsigned c) { const unsigned e = c >> 3, m = c & 7; return e == 0 ? m / 512.0 : (8 + m) * std::ldexp(1.0, (int) e - 10); };
        static const double e2[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
        for (int i : {0, 500}) {
            const int e = exv[(size_t) i], t = tok[(size_t) i];
            const uint8_t* b = host.data() + (size_t) e * blob;
            for (int r : {0, 1, 2, 3, 100, 511, 512, 513}) {   // gate rows, then up rows
                const uint8_t* row = b + (size_t) r * (H / 64) * 36;
                double ref = 0;
                for (int v = 0; v < H; ++v) {
                    const uint8_t* blk = row + (size_t) (v / 64) * 36;
                    const int sub = (v % 64) / 16, j = v % 16;
                    const uint8_t byte = blk[4 + 8 * sub + (j % 8)];
                    ref += e2[j < 8 ? byte & 15 : byte >> 4] * ue(blk[sub]) * x[(size_t) t * H + v];
                }
                std::printf("entry %4d expert %3d token %4d row %3d: mmq %12.5f   host %12.5f\n", i, e, t, r, gu[(size_t) i * 2 * F + r], ref);
            }
        }
    }
    // h in expert order -> compare against the per-pair h by pair index first
    std::vector<float> h_new((size_t) P * F), h_old((size_t) P * F);
    std::vector<int32_t> dst(P);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h_new.data(), dh, h_new.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dst.data(), ddst, (size_t) P * 4, cudaMemcpyDeviceToHost));
    mmq::quantize(dh, nullptr, dmx, kType, F, F, P, nullptr);
    mmq::Product d = p;
    d.w = dev + 2 * (size_t) F * ((size_t) H / 64 * 36);
    d.w_rows = H;
    d.w_cols = F;
    d.ids = ddst;
    d.dst = dy_new;
    d.ld_dst = H;
    ctx.run(d, nullptr);
    CK(cudaDeviceSynchronize());

    // h of the per-pair path (recomputed: dh was overwritten)
    moe::gate_up(ex, dxq, dxs, didx, N, K, dh, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h_old.data(), dh, h_old.size() * 4, cudaMemcpyDeviceToHost));
    double hn = 0, hd = 0;
    for (int i = 0; i < P; ++i)
        for (int r = 0; r < F; ++r) {
            const double a = h_old[(size_t) dst[i] * F + r], b = h_new[(size_t) i * F + r];
            hn += (a - b) * (a - b);
            hd += a * a;
        }
    std::printf("h   : relative RMS difference per-pair vs grouped = %.5f\n", std::sqrt(hn / hd));

    std::vector<float> y_old((size_t) P * H), y_new((size_t) P * H);
    CK(cudaMemcpy(y_old.data(), dy_old, y_old.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y_new.data(), dy_new, y_new.size() * 4, cudaMemcpyDeviceToHost));
    double yn = 0, yd = 0, ymax = 0;
    for (size_t i = 0; i < y_old.size(); ++i) {
        const double a = y_old[i], b = y_new[i];
        yn += (a - b) * (a - b);
        yd += a * a;
        ymax = std::max(ymax, std::fabs(a - b));
    }
    const double rel = std::sqrt(yn / yd);
    std::printf("y   : mmq vs per-pair: relative RMS difference = %.5f, max abs = %.4f (rms of y %.4f)\n", rel, ymax, std::sqrt(yd / y_old.size()));

    // the FP4 block-scaled path, as forward.cpp's moe_apply runs it, for the first n tokens: y by pair index, H16 by entry
    void *da4x, *da4h;
    float *dgsx, *dgsh, *dy_fp4;
    int32_t *dtile, *dblk;
    uint16_t* dh16;
    CK(cudaMalloc(&da4x, fp4::token_bytes(N, 2 * (int64_t) H)));
    CK(cudaMalloc(&da4h, fp4::a4_bytes(P, 2 * (int64_t) F, E)));
    CK(cudaMalloc(&dgsx, (size_t) N * 4));
    CK(cudaMalloc(&dgsh, fp4::scale_count(P, E) * 4));
    CK(cudaMalloc(&dtile, (E + 1) * 4));
    CK(cudaMalloc(&dblk, (E + 1) * 4));
    CK(cudaMalloc(&dh16, (size_t) P * F * 2));
    CK(cudaMalloc(&dy_fp4, (size_t) P * H * 4));
    const auto run_fp4 = [&](int n, std::vector<float>& y) {
        const int pairs = n * K;
        moe::group(didx, n, K, E, dbounds, dtok, ddst, dexp, nullptr);
        fp4::plan_blocks(dbounds, E, dtile, dblk, nullptr);
        fp4::quantize_tokens(dx, n, H, H, da4x, dgsx, nullptr, true);
        fp4::Product g;
        g.base = dev;
        g.stride = blob;
        g.blk_off = dblk;
        g.blocks = fp4::max_blocks(pairs, E);
        g.n = E;
        g.w_rows = 2 * F;
        g.w_cols = H;
        g.a4 = da4x;
        g.gs = dgsx;
        g.bounds = dbounds;
        g.ids = dtok;
        g.tokens = n;
        g.swiglu = true;
        g.tail_off = moe::tail_offset(H, F);
        g.dst_f16 = dh16;
        g.ld_dst = F;
        g.two = true;
        fp4::run(g, nullptr);
        fp4::quantize_f16(dh16, dbounds, dtile, E, da4h, dgsh, F, F, pairs, nullptr);
        fp4::Product dn = g;
        dn.swiglu = false;
        dn.ids = nullptr;
        dn.w_off = 2 * (size_t) F * ((size_t) H / 64 * 36);
        dn.w_rows = H;
        dn.w_cols = F;
        dn.a4 = da4h;
        dn.gs = dgsh;
        dn.tile_off = dtile;
        dn.dst_f16 = nullptr;
        dn.dst = dy_fp4;
        dn.dst_ids = ddst;
        dn.ld_dst = H;
        fp4::run(dn, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        y.resize((size_t) pairs * H);
        cudaMemcpy(y.data(), dy_fp4, y.size() * 4, cudaMemcpyDeviceToHost);
        return true;
    };
    std::vector<float> y_fp4;
    if (!run_fp4(N, y_fp4)) { std::printf("fp4 run failed\n"); return 1; }
    const auto diff = [&](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
        double n2 = 0, d2 = 0, mx = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            n2 += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
            d2 += (double) b[i] * b[i];
            mx = std::max(mx, std::fabs((double) a[i] - b[i]));
        }
        std::printf("y   : fp4 vs %-8s relative RMS difference = %.5f, max abs = %.4f\n", what, std::sqrt(n2 / d2), mx);
        return std::sqrt(n2 / d2);
    };
    const double r_pair = diff(y_fp4, y_old, "per-pair"), r_mmq = diff(y_fp4, y_new, "mmq");

    // a double-precision host reference (dequantize the blob, accumulate in double) of 24 pairs, every column of y
    {
        static const double e2[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
        auto ue = [](unsigned c) { const unsigned e = c >> 3, m = c & 7; return e == 0 ? m / 512.0 : (8 + m) * std::ldexp(1.0, (int) e - 10); };
        const auto wv = [&](const uint8_t* row, int v) {
            const uint8_t* blk = row + (size_t) (v / 64) * 36;
            const int sub = (v % 64) / 16, j = v % 16;
            const uint8_t byte = blk[4 + 8 * sub + (j % 8)];
            return e2[j < 8 ? byte & 15 : byte >> 4] * ue(blk[sub]);
        };
        double fn = 0, fd = 0, fmx = 0, pn = 0;
        const size_t down_off = 2 * (size_t) F * ((size_t) H / 64 * 36);
        for (int q = 0; q < 24; ++q) {
            const int p = (q * 337) % P, t = p / K, e = idx[(size_t) p];
            const uint8_t* b = host.data() + (size_t) e * blob;
            const float* tail = (const float*) (b + moe::tail_offset(H, F));
            std::vector<double> h(F);
            for (int r = 0; r < F; ++r) {
                double g = 0, u = 0;
                for (int v = 0; v < H; ++v) {
                    g += wv(b + (size_t) r * (H / 64) * 36, v) * x[(size_t) t * H + v];
                    u += wv(b + (size_t) (F + r) * (H / 64) * 36, v) * x[(size_t) t * H + v];
                }
                g *= tail[0];
                u *= tail[1];
                h[r] = g / (1.0 + std::exp(-g)) * u * tail[2];
            }
            for (int c = 0; c < H; ++c) {
                double y = 0;
                for (int r = 0; r < F; ++r) y += wv(b + down_off + (size_t) c * (F / 64) * 36, r) * h[r];
                const double a = y_fp4[(size_t) p * H + c], o = y_old[(size_t) p * H + c];
                fn += (a - y) * (a - y);
                pn += (o - y) * (o - y);
                fd += y * y;
                fmx = std::max(fmx, std::fabs(a - y));
            }
        }
        std::printf("y   : vs host double reference (24 pairs): fp4 relative RMS = %.5f (max abs %.4f), per-pair int8 = %.5f\n",
                    std::sqrt(fn / fd), fmx, std::sqrt(pn / fd));
    }

    // a token's rows do not depend on the other tokens: the first half of the tokens alone gives the same bits
    {
        std::vector<float> y_half;
        if (!run_fp4(N / 2, y_half)) { std::printf("fp4 half run failed\n"); return 1; }
        size_t bad = 0;
        for (size_t i = 0; i < y_half.size(); ++i) bad += std::memcmp(&y_half[i], &y_fp4[i], 4) != 0;
        std::printf("bits: %zu of %zu values of the first %d tokens differ between a batch of %d and of %d tokens\n", bad, y_half.size(), N / 2, N, N / 2);
        if (bad != 0) return 1;
    }
    return r_pair < 0.05 && r_mmq < 0.05 ? 0 : 1;
}
