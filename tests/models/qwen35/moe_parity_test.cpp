// tests/models/qwen35/moe_parity_test.cpp - a qwen35moe pack's routed experts: the per-pair INT8 kernels (decode) against the
// grouped MMQ products (prompt chunks), on the first layer's real experts with the same routing.  Usage:
//   qwen35_moe_parity_test PACK.basalt [rows]     (skipped, exit 77, without a pack)
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/moe_ffn.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace moe = basalt::kernels::moe;
namespace mmq = basalt::prefill::mmq;

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
    CK(cudaDeviceSynchronize());
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
    std::printf("y   : relative RMS difference = %.5f, max abs = %.4f (rms of y %.4f)\n", rel, ymax, std::sqrt(yd / y_old.size()));
    return rel < 0.05 ? 0 : 1;
}
