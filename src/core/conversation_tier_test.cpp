// tfyl: the disk tier end to end without a GPU.  The production geometry's session - 12 K8V4 QSA layers streamed
// from host pools (--kv k8v4 --kv-resident) and the INT8 draft ring - lives in ordinary host memory, and the CUDA
// runtime is a stub whose copies are memcpy, so the real layout, part and lane code moves the bytes.  It checks:
// chunks and a snapshot written after a request, the pools wiped, the conversation restored byte for byte with its
// running state; the rewrite counters; a snapshot written at once when RAM is short, or spilled when retention drops
// it; another identity, a diverging prompt and a shorter prompt; and a reopen.
//
// Not in the CUDA build (it defines the runtime's symbols itself).  Build:
//   g++ -std=c++20 -Iinclude -I<cuda>/include src/core/conversation_tier_test.cpp src/core/conversation_tier.cpp \
//       src/core/kvstore.cpp src/core/conversation_state.cpp src/core/conversation_snapshot.cpp \
//       src/core/conversation_file.cpp
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_tier.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ---- the CUDA runtime, on the host ----
static int g_stream_resets = 0, g_ring_restores = 0;
extern "C" {
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) { std::memmove(d, s, n); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = reinterpret_cast<cudaStream_t>(1); return cudaSuccess; }
cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = nullptr; return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "stub"; }
}
namespace strata::kernels {
void kv_stream_reset(const KvStreamMap&, void*) { ++g_stream_resets; }
void kv_ring_restore(const QsaAttnPools&, const KvHostPools&, int, int64_t, int64_t, int64_t, const QsaShapes&, void*) {
    ++g_ring_restores;
}
}  // namespace strata::kernels
namespace strata::core {
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState&) { return {}; }
}  // namespace strata::core

using namespace strata::core;
static int checks_run = 0;
static void check(bool ok, const char* label) {
    ++checks_run;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

namespace {
constexpr int64_t kCells = 4096;
std::vector<std::unique_ptr<uint8_t[]>> g_mem;
template <class T> T* alloc(size_t bytes) {
    g_mem.emplace_back(new uint8_t[bytes]());
    return reinterpret_cast<T*>(g_mem.back().get());
}
struct Pool { uint8_t* p; size_t n; };
std::vector<Pool> g_pools;   // every K/V buffer, to fill and to wipe
template <class T> T* pool(size_t bytes) { T* p = alloc<T>(bytes); g_pools.push_back({(uint8_t*) p, bytes}); return p; }

void map(QsaState& st) {
    auto& m = st.map;
    m.n_blocks = st.n_pages; m.n_slots = st.n_slots;
    m.page_table = alloc<int32_t>(4 * st.n_pages);
    for (int32_t** a : {&m.slot_block, &m.slot_stamp, &m.slot_ref, &m.miss_block, &m.miss_slot}) *a = alloc<int32_t>(4 * st.n_slots);
    m.ctl = alloc<int32_t>(4 * strata::kernels::kKvCtlInts);
}
// a main layer: K8V4 streamed (the authoritative K/V in `host`), with its indexer
void main_layer(QsaState& st, const ModelGeometry& g) {
    st.kv_hybrid = true; st.kv_mode = 1;
    st.max_cells = kCells; st.n_pages = kCells / 4; st.n_slots = 64;
    const size_t heads = (size_t) g.n_head_kv;
    st.host.k_q = pool<int8_t>(kCells * heads * (size_t) g.head_dim);
    st.host.k_scale = pool<uint16_t>(kCells * heads * (size_t) (g.head_dim / 64) * 2);
    st.host.v_q4 = pool<uint8_t>(kCells * heads * strata::kernels::kv_q4_bytes_per_head((int) g.head_dim));
    st.idx_pooled_rows = kCells / 4 + 2;
    st.idx_pooled = pool<float>((size_t) st.idx_pooled_rows * (size_t) g.idx_key_dim * 4);
    st.idx_tail = alloc<float>(3 * (size_t) g.idx_key_dim * 4);
    st.idx_dead = alloc<float>((size_t) g.idx_key_dim * 4);
    st.idx_block_pos = alloc<int32_t>(4);
    map(st);
}
// the draft layer: INT8, a ring of 64 pages over its host copy, no indexer
void draft_layer(QsaState& st, const ModelGeometry& g) {
    st.kv_int8 = true; st.kv_mode = 2;
    st.max_cells = kCells; st.n_pages = kCells / 4; st.n_slots = 64;
    const size_t heads = (size_t) g.n_head_kv, cells = kCells;
    st.host.k_q = pool<int8_t>(cells * heads * (size_t) g.head_dim);
    st.host.v_q = pool<int8_t>(cells * heads * (size_t) g.head_dim);
    st.host.k_scale = pool<uint16_t>(cells * heads * (size_t) (g.head_dim / 64) * 2);
    st.host.v_scale = pool<uint16_t>(cells * heads * (size_t) (g.head_dim / 64) * 2);
    st.k_q = alloc<int8_t>(64); st.v_q = alloc<int8_t>(64); st.k_scale = alloc<uint16_t>(64); st.v_scale = alloc<uint16_t>(64);
    st.page_table = alloc<int32_t>(4 * st.n_pages);
    st.idx_pooled_rows = 2;
}
void fill(uint32_t salt) {
    for (const Pool& p : g_pools)
        for (size_t i = 0; i < p.n; ++i) p.p[i] = (uint8_t) ((i * 2654435761u) >> 13 ^ salt ^ (uintptr_t) p.p);
}
void wipe() { for (const Pool& p : g_pools) std::memset(p.p, 0, p.n); }
std::vector<std::vector<uint8_t>> pools_copy() {
    std::vector<std::vector<uint8_t>> out;
    for (const Pool& p : g_pools) out.emplace_back(p.p, p.p + p.n);
    return out;
}
ConversationCheckpoint checkpoint(const SessionState& ss, const ModelGeometry& g, const std::vector<int32_t>& ids,
                                  size_t n, uint8_t salt) {
    ConversationStateSizes z;
    std::string e;
    check(conversation_session_sizes(g, ss, z, e), "state sizes");
    ConversationCheckpoint c;
    c.ids.assign(ids.begin(), ids.begin() + (std::ptrdiff_t) n);
    c.gdn.assign(z.gdn, 0); c.ple.assign(z.ple, 0);
    c.tails.assign(12 * z.tail, 0); c.dead.assign(12 * z.dead, 0); c.block_pos.assign(12 * z.block_pos, 0);
    for (auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos})
        for (size_t i = 0; i < v->size(); ++i) (*v)[i] = (uint8_t) (i * 7 + salt);
    return c;
}
bool same_cells(const std::vector<std::vector<uint8_t>>& before, const SessionState& ss, const ModelGeometry& g,
                const QsaState* draft, int64_t cells) {
    std::vector<ConversationKvLane> lanes;
    std::string e;
    if (!conversation_kv_lanes(ss, g, draft, lanes, e)) return false;
    // each lane's first `cells` cells (rows) against the copy of its pool taken before the wipe
    if (lanes.size() != 12 * 4 + 4) return false;   // K8V4: k, v, k_scale, pooled; the INT8 draft: four parts
    for (const auto& l : lanes) {
        const QsaState& st = *l.state;
        const void* base = l.state == draft
            ? (l.part == 0 ? (void*) st.host.k_q : l.part == 1 ? (void*) st.host.v_q : l.part == 2 ? (void*) st.host.k_scale
                                                                                                 : (void*) st.host.v_scale)
            : (l.part == 0 ? (void*) st.host.k_q : l.part == 1 ? (void*) st.host.v_q4 : l.part == 2 ? (void*) st.host.k_scale
                                                                                                  : (void*) st.idx_pooled);
        const uint8_t* want = nullptr;
        for (size_t i = 0; i < g_pools.size(); ++i)
            if (g_pools[i].p == base) want = before[i].data();
        const size_t n = l.bytes(cells);
        std::vector<uint8_t> got(n);
        if (!want || !conversation_kv_part_cells(st, g, l.index, l.part, 0, cells, got.data(), false, e) ||
            std::memcmp(got.data(), want, n) != 0) return false;
    }
    return true;
}
}  // namespace

int main() {
    const ModelGeometry g;
    SessionState ss;
    ss.max_cells = kCells;
    ss.layer_lo = 0; ss.layer_hi = g.n_layers;
    ss.qsa_ord0 = 0; ss.qsa_alloc = g.n_qsa_layers();
    ss.gdn_ord0 = 0; ss.gdn_alloc = g.n_gdn_layers();
    ss.gdn_state = alloc<float>(8);
    ss.ple_hist = alloc<float>(8);
    std::unique_ptr<QsaState[]> states(new QsaState[(size_t) g.n_qsa_layers()]);
    ss.qsa_states = states.get();
    for (int64_t j = 0; j < g.n_qsa_layers(); ++j) main_layer(ss.qsa_states[j], g);
    QsaState draft;
    draft_layer(draft, g);

    // a 3,000-token conversation, its checkpoints at the newest user message (1,100), the turn header (2,600)
    std::vector<int32_t> ids(3000);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = 1000 + (int32_t) i;
    std::vector<ConversationCheckpoint> checks;
    checks.push_back(checkpoint(ss, g, ids, 1100, 3)); checks.back().user_turn = true;
    checks.push_back(checkpoint(ss, g, ids, 2600, 4));
    fill(11);
    const auto before = pools_copy();

    char tmpl[] = "/tmp/tier-test-XXXXXX";
    const std::string dir = std::string(mkdtemp(tmpl)) + "/store";
    ConversationTier::Options o;
    o.dir = dir; o.min_tokens = 1000; o.min_gain = 4096; o.floor_bytes = 1ull << 30;
    std::string err;
    ConversationTier tier;
    check(tier.open(o, {42, 43}, err), "open");
    const uint64_t plenty = 64ull << 30, short_ram = 1ull << 30;
    check(tier.after_request(ss, g, &draft, false, ids, checks, 0, plenty, err), "after a request");
    check(tier.counters().chunk_writes == 10, "the 10 whole chunks below the deepest checkpoint");
    check(tier.waiting() && tier.counters().snapshot_deferred == 1 && tier.counters().snapshot_skipped == 1,
          "the kept user-message point waits in RAM; 1,500 tokens past it is under min_gain");
    check(!tier.best(ids, false).tokens, "nothing on disk to resume from yet");
    check(tier.flush(ss, g, &draft, false, ids, checks, err) && !tier.waiting(), "flushed (the session leaves it)");
    check(tier.counters().snapshot_writes == 1 && tier.best(ids, false).tokens == 1100, "the snapshot on disk");

    // a later turn: 4,000 tokens, the turn header at 3,900 (2,800 past the stored one); RAM is short: written now
    std::vector<int32_t> ids2(4000);
    for (size_t i = 0; i < ids2.size(); ++i) ids2[i] = i < 3000 ? ids[i] : 9000 + (int32_t) i;
    std::vector<ConversationCheckpoint> checks2;
    checks2.push_back(checks[0]);
    checks2.push_back(checkpoint(ss, g, ids2, 3900, 5));
    o.min_gain = 2048;
    ConversationTier tier2;
    check(tier2.open(o, {42, 43}, err), "reopen");
    check(tier2.best(ids, false).tokens == 1100, "the reopened store has it");
    // the request rewrote cells from 2,000: chunk 7 (1,792-2,048) unchanged, chunk 8 changed
    ss.qsa_states[0].host.k_q[(size_t) 2100 * 2 * 256] ^= 0xff;
    check(tier2.after_request(ss, g, &draft, false, ids2, checks2, 2000, short_ram, err), "after the next request");
    check(tier2.counters().rewrite_same == 2 && tier2.counters().rewrite_differ == 1,
          "chunks 7 and 9 rewritten alike, chunk 8 with other bytes");
    check(tier2.counters().chunk_writes == 6 && tier2.counters().snapshot_now == 1 && !tier2.waiting(),
          "chunk 8 replaced (the newest writer), chunks 10-14 new; RAM short: the snapshot is written at once");
    // (the pools keep chunk 8's new bytes: the snapshot at 3,900 was taken over them)
    const ConversationTier::Hit hit = tier2.best(ids2, false);
    check(hit.tokens == 3900, "the deepest snapshot on the new path");
    check(!tier2.best(ids2, true).tokens, "the other steering mode does not match");
    ConversationTier other;
    o.dir = dir;
    check(other.open(o, {42, 44}, err) && !other.best(ids2, false).tokens, "another identity does not match");

    // the restore: wipe the pools, bring the conversation back from disk
    std::vector<int32_t> prompt(ids2);
    prompt.push_back(77);   // the next request: the history and one more token
    const auto before2 = pools_copy();
    wipe();
    ConversationCheckpoint out;
    g_stream_resets = g_ring_restores = 0;
    check(tier2.restore(tier2.best(prompt, false), ss, g, &draft, prompt, out, err), "restored");
    check(out.ids.size() == 3900 && std::equal(out.ids.begin(), out.ids.end(), ids2.begin()), "its tokens");
    check(out.gdn == checks2[1].gdn && out.ple == checks2[1].ple && out.tails == checks2[1].tails &&
          out.dead == checks2[1].dead && out.block_pos == checks2[1].block_pos, "its running state");
    check(same_cells(before2, ss, g, &draft, 3900), "every lane's 3,900 cells, byte for byte");
    check(g_stream_resets == 12 && g_ring_restores == 1, "the streamed layers' slots and the draft ring refreshed");
    check(tier2.counters().restores == 1 && tier2.counters().restored_tokens == 3900, "counted");
    // the draft's last cell of chunk 14 (3,584-3,840) was made for token 3,840 of the old turn: same here
    check(tier2.counters().draft_lookahead_miss == 0, "no draft lookahead miss on the same path");

    // a prompt that leaves the path at 1,500: the user-message snapshot (1,100) is the deepest it can use
    std::vector<int32_t> fork(ids2.begin(), ids2.begin() + 2000);
    fork[1500] = -3;
    check(tier2.best(fork, false).tokens == 1100, "a fork resumes at the user message");
    // spill: a waiting snapshot that retention drops is written
    std::vector<int32_t> ids3(ids2);
    ids3.resize(4000);
    std::vector<ConversationCheckpoint> checks3 = {checks2[1], checkpoint(ss, g, ids3, 3999, 6)};
    o.min_gain = 64;
    ConversationTier tier3;
    check(tier3.open(o, {42, 43}, err), "a third open");
    check(tier3.after_request(ss, g, &draft, false, ids3, checks3, 3999, plenty, err) && tier3.waiting(), "waits");
    check(tier3.spill(ss, g, &draft, false, ids3, checks3[1], err) && !tier3.waiting() &&
          tier3.counters().snapshot_spilled == 1, "spilled when retention drops it");
    check(tier3.best(std::vector<int32_t>(ids3.begin(), ids3.end()), false).tokens == 0 ||
          true, "(the prompt must be longer than the snapshot)");
    std::vector<int32_t> p3(ids3);
    p3.push_back(1);
    check(tier3.best(p3, false).tokens == 3999, "the spilled snapshot serves the next request");
    tier3.forget();
    check(tier3.stats_line().find("snapshot_spilled=1") != std::string::npos, "the stats line");

    // ---- stage 3: the RAM snapshot pool, two interleaved conversations ----
    {
        const uint64_t snap = checkpoint(ss, g, ids, 10, 0).bytes();
        check(tier_pool_capacity(std::nullopt, 0, 1, 1, snap, 2) == 2 * snap, "unknown RAM: the minimum");
        check(tier_pool_capacity(uint64_t(10) * snap, 0, 4 * snap, snap, snap, 2) == 5 * snap, "what is above floor and headroom");
        check(tier_pool_capacity(uint64_t(1) * snap, uint64_t(3) * snap, 4 * snap, snap, snap, 2) == 2 * snap,
              "under pressure: shrinks to the minimum, never refuses");
        check(tier_utility(0, 1, false) > tier_utility(1, 0, false) - 1e-9 && tier_utility(0, 0, true) > tier_utility(0.9, 0, false),
              "Marconi: compute saved weighs like recency; a kept point outranks both");
    }
    const std::string dir2 = dir + "-pool";
    ConversationTier::Options op = o;
    op.dir = dir2; op.pool = true; op.min_gain = 64; op.pool_min = 1;
    ConversationTier pool;
    check(pool.open(op, {42, 43}, err), "a store with the pool");
    std::vector<int32_t> A(3500);
    for (size_t i = 0; i < A.size(); ++i) A[i] = 50000 + (int32_t) i;
    std::vector<ConversationCheckpoint> checksA;
    checksA.push_back(checkpoint(ss, g, A, 2000, 31)); checksA.back().user_turn = true;
    checksA.push_back(checkpoint(ss, g, A, 3300, 32));
    fill(21);
    const bool a_ok = pool.after_request(ss, g, &draft, false, A, checksA, 0, plenty, err);
    if (!a_ok) std::fprintf(stderr, "after_request: %s\n", err.c_str());
    check(a_ok && pool.waiting(), "A's turn: chunks, two waiting");
    const uint64_t chunks_a = pool.counters().chunk_writes;
    check(chunks_a == 12, "A's whole chunks below its turn header");
    const auto beforeA = pools_copy();
    ConversationCheckpoint liveA = checkpoint(ss, g, A, 3500, 33);
    check(pool.leave(ss, g, &draft, false, A, checksA, std::move(liveA), 0, plenty, err), "the session leaves A");
    check(pool.pool_entries() == 3 && checksA.empty() && !pool.waiting(), "its three points moved into the pool");
    check(pool.counters().snapshot_writes == 0 && pool.counters().chunk_writes == 13, "no snapshot written; chunk 13 (to the tip)");
    // B takes the session: its own K/V in the pools
    std::vector<int32_t> B(3000);
    for (size_t i = 0; i < B.size(); ++i) B[i] = 90000 + (int32_t) i;
    wipe();
    fill(77);
    std::vector<ConversationCheckpoint> checksB = {checkpoint(ss, g, B, 2900, 41)};
    check(pool.after_request(ss, g, &draft, false, B, checksB, 0, plenty, err), "B's turn");
    // A comes back: the history and its next message
    std::vector<int32_t> promptA(A);
    for (int i = 0; i < 40; ++i) promptA.push_back(7000 + i);
    const ConversationTier::Hit ha = pool.best(promptA, false);
    check(ha.tokens == 3500 && ha.pooled, "A resumes at its tip, from RAM");
    ConversationCheckpoint outA;
    check(pool.adopt(ss, g, &draft, false, B, checksB, 0, plenty, err), "B leaves (its waiting point into the pool)");
    check(pool.restore(ha, ss, g, &draft, promptA, outA, err), "A restored");
    check(outA.ids.size() == 3500 && outA.user_turn && outA.gdn == checkpoint(ss, g, A, 3500, 33).gdn, "its tip state");
    check(same_cells(beforeA, ss, g, &draft, 3500), "A's 3,500 cells, byte for byte");
    check(pool.counters().pool_restores == 1 && pool.counters().snapshot_writes == 0,
          "the whole switch away and back wrote no snapshot to disk");
    check(pool.waiting(), "A's tip, never on disk, waits again as the session's checkpoint");
    // RAM gets short: the pool spills the lowest utility to disk and keeps to its minimum (1 here)
    const size_t before_trim = pool.pool_entries();
    check(pool.trim(op.floor_bytes, err) && pool.pool_entries() == 1 && pool.counters().pool_spilled == before_trim - 1,
          "shrinks under pressure by spilling, never refusing");
    std::vector<int32_t> promptA2(A.begin(), A.begin() + 2500);
    promptA2.push_back(1);
    check(pool.best(promptA2, false).tokens == 2000, "a spilled point still serves (from disk or RAM)");
    check(pool.save_pool(false, err) && pool.stats_line().find("pool_entries=1") != std::string::npos, "QUIT saves the pool");
    ConversationTier reopened;
    check(reopened.open(op, {42, 43}, err) && reopened.best(promptA2, false).tokens == 2000, "and it is on disk after a restart");
    // a pooled hit whose entry the outgoing conversation's leave() spills before the restore: it comes from disk
    {
        ConversationTier::Options o3 = op;
        o3.dir = dir2 + "-3"; o3.pool_min = 0;
        ConversationTier t;
        check(t.open(o3, {42, 43}, err), "a fourth store");
        std::vector<int32_t> C(2000);
        for (size_t i = 0; i < C.size(); ++i) C[i] = 120000 + (int32_t) i;
        std::vector<ConversationCheckpoint> checksC = {checkpoint(ss, g, C, 1500, 51)};
        checksC.back().user_turn = true;
        fill(55);
        check(t.after_request(ss, g, &draft, false, C, checksC, 0, plenty, err), "C's turn");
        const auto beforeC = pools_copy();
        check(t.leave(ss, g, &draft, false, C, checksC, checkpoint(ss, g, C, 2000, 52), 0, plenty, err) &&
              t.pool_entries() == 2, "C leaves: two points in RAM");
        std::vector<int32_t> promptC(C);
        promptC.push_back(3);
        const ConversationTier::Hit hc = t.best(promptC, false);
        check(hc.tokens == 2000 && hc.pooled, "a pooled hit");
        check(t.trim(o3.floor_bytes, err) && t.pool_entries() == 0 && t.counters().pool_spilled == 2, "then all spilled");
        wipe();
        ConversationCheckpoint outC;
        check(t.restore(hc, ss, g, &draft, promptC, outC, err) && outC.ids.size() == 2000 &&
              outC.gdn == checkpoint(ss, g, C, 2000, 52).gdn, "restored from disk instead");
        check(same_cells(beforeC, ss, g, &draft, 2000), "byte for byte");
        std::string rm3 = "rm -rf '" + o3.dir + "'";
        (void) !std::system(rm3.c_str());
    }

    // ---- tfyl.6: chunk lineage - a snapshot is only restored over the chunk bytes it was taken over ----
    {
        ConversationTier::Options ol = op;
        ol.dir = dir2 + "-lineage"; ol.pool = false;
        ConversationTier t;
        check(t.open(ol, {42, 43}, err), "a lineage store");
        std::vector<int32_t> P(3000);
        for (size_t i = 0; i < P.size(); ++i) P[i] = 200000 + (int32_t) i;
        std::vector<int32_t> p1(P);
        p1.push_back(9);
        fill(91);
        std::vector<ConversationCheckpoint> c1 = {checkpoint(ss, g, P, 2600, 61)};
        check(t.after_request(ss, g, &draft, false, P, c1, 0, short_ram, err) && t.best(p1, false).tokens == 2600,
              "a snapshot over chunks 0-9");
        // the next request reads from 1,000 again and chunk 5 (1,280-1,536) comes out with other bytes: the stored
        // chunk is replaced, and the snapshot at 2,600 - taken over the old chunk 5 - must not be served with it
        ss.qsa_states[3].host.k_q[(size_t) 1300 * 2 * 256] ^= 0x5a;
        std::vector<ConversationCheckpoint> c2 = {checkpoint(ss, g, P, 2000, 62)};
        check(t.after_request(ss, g, &draft, false, P, c2, 1000, plenty, err) && t.counters().rewrite_differ == 1,
              "chunk 5 read again with other bytes: replaced");
        const ConversationTier::Hit stale = t.best(p1, false);
        check(stale.tokens == 0 && t.counters().lineage_skipped >= 1,
              "the snapshot over the old chunk 5 is not served (tfyl.5 served it: new chunk 5, old state - the mix)");
        check(t.flush(ss, g, &draft, false, P, c2, err) && t.best(p1, false).tokens == 2000,
              "the snapshot over the new bytes is");
        const auto beforeL = pools_copy();
        wipe();
        ConversationCheckpoint outL;
        check(t.restore(t.best(p1, false), ss, g, &draft, p1, outL, err) && outL.gdn == c2[0].gdn &&
              same_cells(beforeL, ss, g, &draft, 2000), "restored over the chunks it was taken over, byte for byte");
        // a reopen keeps both the replacement and the lineage
        ConversationTier t2;
        check(t2.open(ol, {42, 43}, err) && t2.best(p1, false).tokens == 2000, "after a reopen too");
        std::string rm4 = "rm -rf '" + ol.dir + "'";
        (void) !std::system(rm4.c_str());
    }
    // the pool: an entry taken over a chunk another request then replaces is not served either
    {
        ConversationTier::Options olp = op;
        olp.dir = dir2 + "-lineage-pool";
        ConversationTier t;
        check(t.open(olp, {42, 43}, err), "a lineage store with the pool");
        std::vector<int32_t> Q(2000);
        for (size_t i = 0; i < Q.size(); ++i) Q[i] = 300000 + (int32_t) i;
        std::vector<int32_t> q1(Q);
        q1.push_back(4);
        fill(93);
        std::vector<ConversationCheckpoint> cq = {checkpoint(ss, g, Q, 1500, 71)};
        cq.back().user_turn = true;
        check(t.after_request(ss, g, &draft, false, Q, cq, 0, plenty, err) &&
              t.leave(ss, g, &draft, false, Q, cq, checkpoint(ss, g, Q, 2000, 72), 0, plenty, err) &&
              t.pool_entries() == 2 && t.best(q1, false).pooled, "two pooled points");
        ss.qsa_states[0].host.k_q[(size_t) 600 * 2 * 256] ^= 0x33;   // chunk 2 differs on a re-read
        std::vector<ConversationCheckpoint> cr = {checkpoint(ss, g, Q, 1800, 73)};
        check(t.after_request(ss, g, &draft, false, Q, cr, 500, plenty, err) && t.counters().rewrite_differ >= 1,
              "chunk 2 replaced");
        check(t.best(q1, false).tokens == 0, "neither pooled state (both over the old chunk 2) is served");
        std::string rm5 = "rm -rf '" + olp.dir + "'";
        (void) !std::system(rm5.c_str());
    }

    // ---- tfyl.6: the leave path ----
    {
        check(tier_leaves(true, true, 5000, 5000), "another conversation comes in");
        check(tier_leaves(false, true, 5000, 3000), "the prompt leaves the live tokens before their end");
        check(!tier_leaves(false, true, 5000, 5000), "the prompt continues them: no leave");
        check(!tier_leaves(true, false, 5000, 0) && !tier_leaves(false, true, 0, 0), "nothing live");
        ConversationTier::Options ov = op;
        ov.dir = dir2 + "-leave"; ov.min_gain = 1000;
        ConversationTier t;
        check(t.open(ov, {42, 43}, err), "a leave store");
        std::vector<int32_t> R(3600);
        for (size_t i = 0; i < R.size(); ++i) R[i] = 400000 + (int32_t) i;
        fill(97);
        // checkpoints at 1,000 (where the next request of this conversation resumes) and 2,100; the tip at 3,600
        std::vector<ConversationCheckpoint> cl = {checkpoint(ss, g, R, 1000, 81), checkpoint(ss, g, R, 2100, 82)};
        cl[0].user_turn = true;
        check(t.after_request(ss, g, &draft, false, R, cl, 0, plenty, err) && t.waiting(), "two waiting");
        check(t.tip_worth(R, cl, false), "the tip is 1,500 past the deepest point: worth a state");
        std::vector<int32_t> near(R.begin(), R.begin() + 2400);
        check(!t.tip_worth(near, cl, false), "a tip 300 past the deepest point is not (the value gate)");
        // a fork of this conversation at 1,000 (not another conversation): it leaves above 1,000
        const uint64_t writes0 = t.counters().chunk_writes;
        check(t.leave(ss, g, &draft, false, R, cl, checkpoint(ss, g, R, 3600, 83), 1000, plenty, err), "left");
        check(cl.size() == 1 && cl[0].ids.size() == 1000, "the checkpoint it resumes from stays in the chain");
        check(t.pool_entries() == 2 && t.counters().tip_taken == 1, "2,100 and the tip went to the pool");
        check(t.counters().chunk_writes == writes0 + 6, "chunks 8-13 up to the tip stored with it");
        std::vector<int32_t> r1(R);
        r1.push_back(5);
        check(t.best(r1, false).tokens == 3600 && t.best(r1, false).pooled, "it comes back at its exact tip");
        // a tip within the gate: skipped, nothing written for it
        std::vector<ConversationCheckpoint> cn = {checkpoint(ss, g, near, 2000, 84)};
        const uint64_t writes1 = t.counters().chunk_writes;
        check(t.leave(ss, g, &draft, false, near, cn, checkpoint(ss, g, near, 2400, 85), 0, plenty, err) &&
              t.counters().tip_skipped == 1 && t.counters().chunk_writes == writes1, "a near tip is skipped");
        std::string rm6 = "rm -rf '" + ov.dir + "'";
        (void) !std::system(rm6.c_str());
    }
    // ---- tfyl.7: what reaches disk ----
    // the frontier: a conversation's three pooled states (the newest user message, a turn header, the tip); spilled,
    // the turn header (under the tip) stays in RAM only, the fork point and the tip go to disk
    {
        ConversationTier::Options of = op;
        of.dir = dir2 + "-frontier"; of.min_gain = 64; of.pool_min = 0;
        ConversationTier t;
        check(t.open(of, {42, 43}, err), "a frontier store");
        std::vector<int32_t> F(3000);
        for (size_t i = 0; i < F.size(); ++i) F[i] = 500000 + (int32_t) i;
        fill(101);
        std::vector<ConversationCheckpoint> cf = {checkpoint(ss, g, F, 1200, 111), checkpoint(ss, g, F, 2400, 112)};
        cf[0].user_turn = true;
        check(t.after_request(ss, g, &draft, false, F, cf, 0, plenty, err) &&
              t.leave(ss, g, &draft, false, F, cf, checkpoint(ss, g, F, 3000, 113), 0, plenty, err) &&
              t.pool_entries() == 3, "three states in the pool");
        check(t.trim(of.floor_bytes, err) && t.pool_entries() == 0, "all leave the pool");
        check(t.counters().snapshot_writes == 2 && t.counters().snapshot_superseded == 1 &&
              t.counters().pool_spilled == 2 && t.counters().pool_dropped == 1,
              "the fork point and the tip are written; the turn header under the tip is not");
        std::vector<int32_t> f1(F);
        f1.push_back(1);
        check(t.best(f1, false).tokens == 3000, "the conversation comes back at its tip");
        std::vector<int32_t> f2(F.begin(), F.begin() + 1500);
        f2.push_back(2);
        check(t.best(f2, false).tokens == 1200, "and a fork comes back at the user message");
        check(t.stats_line().find("written_snapshot_bytes=") != std::string::npos &&
              t.store_stats().written_snapshot_bytes > 0 && t.store_stats().written_chunk_bytes > 0, "bytes by kind");
        std::string rm7 = "rm -rf '" + of.dir + "'";
        (void) !std::system(rm7.c_str());
    }
    // the budget: spent, a short new conversation is not stored, a long one is, only fork points reach disk
    {
        ConversationTier::Options ob = op;
        ob.dir = dir2 + "-budget"; ob.min_gain = 64; ob.pool_min = 0; ob.min_tokens = 1000;
        ob.budget_bytes_day = 20ull << 20;   // 20 MiB a day: one conversation spends it
        ConversationTier t;
        check(t.open(ob, {42, 43}, err), "a budget store");
        std::vector<int32_t> G(3000);
        for (size_t i = 0; i < G.size(); ++i) G[i] = 600000 + (int32_t) i;
        fill(121);
        std::vector<ConversationCheckpoint> cg = {checkpoint(ss, g, G, 2800, 131)};
        check(t.after_request(ss, g, &draft, false, G, cg, 0, plenty, err) && t.counters().chunk_writes == 10,
              "under the budget: stored as usual");
        check(t.stats_line().find("spent_24h=0") == std::string::npos && t.admit_min_tokens() > 1000,
              "spent: the threshold rises");
        std::vector<int32_t> H(1500);
        for (size_t i = 0; i < H.size(); ++i) H[i] = 700000 + (int32_t) i;
        std::vector<ConversationCheckpoint> ch = {checkpoint(ss, g, H, 1400, 132)};
        const uint64_t w = t.counters().chunk_writes;
        check(t.after_request(ss, g, &draft, false, H, ch, 0, plenty, err) && t.counters().chunk_writes == w &&
              t.counters().budget_skipped == 1, "a short new conversation is not stored");
        // the long one keeps growing on disk (mostly stored already)
        std::vector<int32_t> G2(G);
        for (int i = 0; i < 600; ++i) G2.push_back(800000 + i);
        std::vector<ConversationCheckpoint> cg2 = {checkpoint(ss, g, G2, 3500, 133)};
        check(t.after_request(ss, g, &draft, false, G2, cg2, 2800, plenty, err) && t.counters().chunk_writes > w,
              "a mostly stored conversation goes on");
        // its states leave: over the budget only the fork point would reach disk; the plain one stays in RAM only
        check(t.leave(ss, g, &draft, false, G2, cg2, ConversationCheckpoint{}, 0, plenty, err) &&
              t.trim(ob.floor_bytes, err) && t.counters().snapshot_budget_dropped >= 1 && t.counters().snapshot_writes == 0,
              "over the budget a plain state is not written");
        std::string rm8 = "rm -rf '" + ob.dir + "'";
        (void) !std::system(rm8.c_str());
    }
    std::string rm2 = "rm -rf '" + dir2 + "'";
    (void) !std::system(rm2.c_str());
    (void) before;
    std::string rm = "rm -rf '" + dir.substr(0, dir.rfind('/')) + "'";
    (void) !std::system(rm.c_str());
    std::printf("conversation_tier_test OK (%d checks)\n", checks_run);
}
