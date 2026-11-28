// tests/models/qwen35/drafter_batch_test.cpp - the MTP drafter run for several sessions as one pass (draft_batch) leaves each
// session's drafts and their chances as the same drafter run for that session alone (draft_begin) does, round after round
// with windows of 1 to 3 rows, some of whose drafts are wrong, so the drafter's history takes both the waiting row and 1 to
// 3 kept rows.  Group A drafts as a batch; group B, the same prompts and the same windows, one session at a time.  A third
// group runs the prompts plain to give the windows their right tokens.  Needs a pack with an MTP layer (argument or
// BASALT_QWEN35MOE_PACK); exits 77 without one.
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "models/qwen35/adapter.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace basalt;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
[[noreturn]] void die(const char* what, const std::string& msg) {
    std::fprintf(stderr, "%s: %s\n", what, msg.c_str());
    std::exit(1);
}
void must(const model::Result& r, const char* what) {
    if (!r.ok()) die(what, r.message);
}
}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("BASALT_QWEN35MOE_PACK");
    const std::string pack = argc > 1 ? argv[1] : env ? env : "";
    if (pack.empty()) {
        std::printf("skip: no pack (argument or BASALT_QWEN35MOE_PACK)\n");
        return 77;
    }
    constexpr int S = 4, kRounds = 12, kPrompt = 300, kOracle = 80;
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) die(pack.c_str(), err);
    const std::unique_ptr<model::ModelAdapter> adapter = models::qwen35::make_adapter();
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = 3 * S * 2048;
    req.sessions = 3 * S;
    req.draft = true;
    req.spec = 8;
    req.kv = "int8";
    model::Started started;
    must(adapter->start(req, started), "start");
    if ((int) started.sessions.size() < 3 * S) die("sessions", "the model gave " + std::to_string(started.sessions.size()));
    if (started.sessions[0]->max_window() < 4) {
        std::printf("skip: the pack has no drafter\n");
        return 77;
    }
    std::vector<model::ModelSession*> A, B, O;
    for (int i = 0; i < S; ++i) {
        A.push_back(started.sessions[(size_t) i].get());
        B.push_back(started.sessions[(size_t) (S + i)].get());
        O.push_back(started.sessions[(size_t) (2 * S + i)].get());
    }
    for (auto& s : started.sessions) must(s->prepare(), "prepare");

    // S prompts of random tokens, each read by its three sessions
    std::vector<std::vector<int32_t>> prompt((size_t) S, std::vector<int32_t>(kPrompt));
    uint32_t x = 12345;
    for (auto& p : prompt)
        for (int32_t& t : p) {
            x = x * 1664525u + 1013904223u;
            t = 1000 + (int32_t) ((x >> 8) % 30000);
        }
    const auto read = [&](model::ModelSession* s, const std::vector<int32_t>& p) {
        model::PrefillRequest r;
        r.tokens = p.data();
        r.n_tokens = kPrompt - 1;
        r.total_tokens = kPrompt - 1;
        must(s->prefill_chunk(r), "prefill");
    };
    for (int i = 0; i < S; ++i) {
        read(A[(size_t) i], prompt[(size_t) i]);
        read(B[(size_t) i], prompt[(size_t) i]);
        read(O[(size_t) i], prompt[(size_t) i]);
    }
    // the plain greedy trajectory of each prompt: traj[i][0] is its last token, then each pick follows the one before
    std::vector<std::vector<int32_t>> traj((size_t) S);
    for (int i = 0; i < S; ++i) {
        traj[(size_t) i].push_back(prompt[(size_t) i].back());
        for (int k = 0; k < kOracle; ++k) {
            int32_t next = -1;
            model::DecodeRequest r;
            r.tokens = &traj[(size_t) i].back();
            r.n_tokens = 1;
            r.pos = kPrompt - 1 + k;
            r.next_token = &next;
            must(O[(size_t) i]->decode_window(r), "oracle");
            traj[(size_t) i].push_back(next);
        }
    }

    std::vector<int> consumed((size_t) S, 0);   // trajectory tokens each pair of sessions has taken
    std::vector<int> t_kept((size_t) S);
    size_t same_ids = 0, same_conf = 0, total = 0;
    int rows_seen[5] = {};
    for (int round = 0; round < kRounds; ++round) {
        std::vector<int32_t> anchors((size_t) S);
        std::vector<int64_t> pos((size_t) S);
        std::vector<int> want((size_t) S, 2), kept_a((size_t) S);
        for (int i = 0; i < S; ++i) {
            // the window: the pair's next token, then 0 to 2 of the right ones after it, the first of them wrong now and then
            const int c = consumed[(size_t) i];
            const int T = 1 + (round + i) % 3;
            std::vector<int32_t> win((size_t) T);
            for (int j = 0; j < T; ++j) win[(size_t) j] = traj[(size_t) i][(size_t) (c + j)];
            if (T > 1 && (round + i) % 4 == 3) win[1] += 1;
            int a = 0;
            std::vector<int32_t> pick_a((size_t) T), pick_b((size_t) T);
            for (int g = 0; g < 2; ++g) {
                model::ModelSession* s = g == 0 ? A[(size_t) i] : B[(size_t) i];
                model::DecodeRequest r;
                r.tokens = win.data();
                r.n_tokens = T;
                r.pos = kPrompt - 1 + c;
                r.row_tokens = g == 0 ? pick_a.data() : pick_b.data();
                must(s->decode_window(r), "window");
            }
            check(pick_a == pick_b, "the two groups pick the same tokens (round " + std::to_string(round) + ", session " + std::to_string(i) + ")");
            while (a < T - 1 && win[(size_t) a + 1] == pick_b[(size_t) a]) ++a;
            t_kept[(size_t) i] = a + 1;
            kept_a[(size_t) i] = a + 1;
            consumed[(size_t) i] = c + a + 1;
            anchors[(size_t) i] = pick_b[(size_t) a];
            pos[(size_t) i] = kPrompt - 1 + consumed[(size_t) i];
            ++rows_seen[a + 1];
            model::ModelSession* sb = B[(size_t) i];
            must(sb->decode_kept(&sb, &t_kept[(size_t) i], 1), "kept B");
        }
        must(A[0]->decode_kept(A.data(), kept_a.data(), S), "kept A");
        // group A as one pass, read to its end; then group B one session at a time (the sessions share the model's window
        // scratch, so one drafting finishes before the next begins, as the server's rounds do)
        int32_t ids_a[S][8] = {};
        float conf_a[S][8] = {};
        int na_all[S] = {};
        must(A[0]->draft_batch(A.data(), anchors.data(), pos.data(), want.data(), S), "draft_batch");
        for (int i = 0; i < S; ++i) must(A[(size_t) i]->draft_end(ids_a[i], conf_a[i], 7, na_all[i], {}), "draft_end A");
        for (int i = 0; i < S; ++i) {
            int32_t ids_b[8] = {};
            float conf_b[8] = {};
            int na = na_all[i], nb = 0;
            must(B[(size_t) i]->draft_begin(anchors[(size_t) i], pos[(size_t) i], want[(size_t) i]), "draft_begin");
            must(B[(size_t) i]->draft_end(ids_b, conf_b, 7, nb, {}), "draft_end B");
            const bool ids = na == nb && std::memcmp(ids_a[i], ids_b, sizeof(int32_t) * (size_t) nb) == 0;
            const bool conf = na == nb && std::memcmp(conf_a[i], conf_b, sizeof(float) * (size_t) nb) == 0;
            same_ids += ids;
            same_conf += conf;
            ++total;
            if (!ids || !conf) std::printf("round %d session %d: drafts %d vs %d, ids %s, chances %s\n", round, i, na, nb, ids ? "same" : "differ", conf ? "same" : "differ");
        }
    }
    std::printf("windows kept 1 / 2 / 3 rows: %d / %d / %d\n", rows_seen[1], rows_seen[2], rows_seen[3]);
    check(rows_seen[1] > 0 && rows_seen[2] > 0 && rows_seen[3] > 0, "the rounds cover windows that keep 1, 2 and 3 rows");
    check(same_ids == total, "the batch drafts the same tokens as the sessions alone (" + std::to_string(same_ids) + " of " + std::to_string(total) + ")");
    check(same_conf == total, "and the same chances to the bit (" + std::to_string(same_conf) + " of " + std::to_string(total) + ")");
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
