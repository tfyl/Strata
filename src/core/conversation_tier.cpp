// tfyl: the conversation cache's disk tier as the serve loop uses it (conversation_tier.hpp).
#include "strata/core/conversation_tier.hpp"
#include "strata/core/conversation_snapshot.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

namespace strata::core {
namespace {
constexpr int64_t kChunk = kvstore::kChunkTokens;
using Clock = std::chrono::steady_clock;

bool fail(std::string& error, const std::string& what) {
    error = "kv store: " + what;
    return false;
}
bool device_idle(std::string& error) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) return true;
    return fail(error, std::string("device synchronize: ") + cudaGetErrorString(e));
}
size_t chunk_bytes(const std::vector<ConversationKvLane>& lanes, int64_t cells) {
    size_t n = 0;
    for (const auto& l : lanes) n += l.bytes(cells);
    return n;
}
/// The cells a tail of `tokens` tokens covers: whole pages and indexer rows.
int64_t tail_cells(const std::vector<ConversationKvLane>& lanes, int64_t tokens, int64_t page) {
    int64_t unit = std::max<int64_t>(page, 1);
    for (const auto& l : lanes)
        if (l.row_cells > unit) unit = l.row_cells;
    return tokens <= 0 ? 0 : (tokens + unit - 1) / unit * unit;
}
/// The lanes' regions in a staging buffer holding `cells` cells of each, lane after lane.
std::vector<uint8_t*> regions(const std::vector<ConversationKvLane>& lanes, uint8_t* base, int64_t cells) {
    std::vector<uint8_t*> r;
    r.reserve(lanes.size());
    for (const auto& l : lanes) { r.push_back(base); base += l.bytes(cells); }
    return r;
}
bool lanes_copy(const std::vector<ConversationKvLane>& lanes, const ModelGeometry& g, int64_t cell0, int64_t cells,
                const std::vector<uint8_t*>& at, bool to_pool, std::string& error) {
    for (size_t i = 0; i < lanes.size(); ++i)
        if (!conversation_kv_part_cells(*lanes[i].state, g, lanes[i].index, lanes[i].part, cell0, cells, at[i], to_pool,
                                        error)) return false;
    return true;
}
int64_t page_of(const std::vector<ConversationKvLane>& lanes, const ModelGeometry& g) {
    ConversationKvStrides s;
    std::string e;
    return !lanes.empty() && conversation_kv_strides(*lanes[0].state, g, lanes[0].index, s, e) ? s.page : 4;
}
}  // namespace

#if defined(_WIN32)
ConversationTier::~ConversationTier() { _aligned_free(staging_); }
#else
ConversationTier::~ConversationTier() { std::free(staging_); }
#endif

bool ConversationTier::open(const Options& o, const std::vector<uint64_t>& identity, std::string& error) {
    opt_ = o;
    identity_ = identity;
    kvstore::Store::Options so;
    so.dir = o.dir;
    so.max_bytes = o.max_bytes;
    if (!store_.open(so, error)) return false;
    return ensure_staging(error);
}

bool ConversationTier::ensure_staging(std::string& error) {
    if (staging_) return true;
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(opt_.staging_bytes, kvstore::kPage);
    if (!p) return fail(error, "no memory for the staging buffer");
#else
    if (posix_memalign(&p, kvstore::kPage, opt_.staging_bytes) != 0) return fail(error, "no memory for the staging buffer");
#endif
    staging_ = static_cast<uint8_t*>(p);
    return true;
}

kvstore::Key ConversationTier::seed(bool cvec) const {
    std::vector<uint64_t> id = identity_;
    id.push_back(cvec ? 1 : 0);
    id.push_back((uint64_t) kChunk);
    return kvstore::chain_seed(id);
}

int64_t ConversationTier::stored_on_path(const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& ids,
                                         int64_t tokens) const {
    const kvstore::Record* r = store_.best_snapshot(keys, ids.data(), (size_t) std::min<int64_t>(tokens + 1, (int64_t) ids.size()));
    return r ? r->tokens : 0;
}

bool ConversationTier::write_chunks(const SessionState& ss, const ModelGeometry& g, const QsaState* draft,
                                    const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& ids,
                                    int64_t upto, int64_t rewrite_from, std::string& error) {
    std::vector<ConversationKvLane> lanes;
    if (!conversation_kv_lanes(ss, g, draft, lanes, error)) return false;
    const size_t per_chunk = chunk_bytes(lanes, kChunk);
    if (!per_chunk) return true;
    const int64_t batch = std::max<int64_t>(1, (int64_t) (opt_.staging_bytes / per_chunk));
    const int64_t whole = std::min<int64_t>(upto / kChunk, (int64_t) keys.size() - 1);
    auto wanted = [&](int64_t j) {
        const kvstore::Record* r = store_.chunk(keys[(size_t) j + 1]);
        return !r || (j + 1) * kChunk > rewrite_from;
    };
    for (int64_t j0 = 0; j0 < whole;) {
        if (!wanted(j0)) { ++c_.chunk_dedup; store_.touch(*store_.chunk(keys[(size_t) j0 + 1])); ++j0; continue; }
        int64_t m = 1;
        while (j0 + m < whole && m < batch && wanted(j0 + m)) ++m;
        const auto at = regions(lanes, staging_, m * kChunk);
        if (!lanes_copy(lanes, g, j0 * kChunk, m * kChunk, at, false, error)) return false;
        for (int64_t k = 0; k < m; ++k) {
            const int64_t j = j0 + k;
            std::vector<kvstore::Piece> pieces;
            pieces.reserve(lanes.size());
            for (size_t i = 0; i < lanes.size(); ++i)
                pieces.push_back({at[i] + (size_t) k * lanes[i].bytes(kChunk), lanes[i].bytes(kChunk)});
            if (const kvstore::Record* r = store_.chunk(keys[(size_t) j + 1])) {
                // a chunk this request rewrote: the first writer stays; did the bytes change?
                (kvstore::digest(pieces) == r->sum ? c_.rewrite_same : c_.rewrite_differ) += 1;
                store_.touch(*r);
                continue;
            }
            const int32_t next = (j + 1) * kChunk < (int64_t) ids.size() ? ids[(size_t) ((j + 1) * kChunk)] : -1;
            bool written = false;
            if (!store_.put_chunk(keys[(size_t) j + 1], keys[(size_t) j], next, pieces, error, &written)) return false;
            if (written) ++c_.chunk_writes;
        }
        j0 += m;
    }
    return true;
}

bool ConversationTier::write_snapshot(const SessionState& ss, const ModelGeometry& g, const QsaState* draft,
                                      const std::vector<kvstore::Key>& keys, const ConversationCheckpoint& c,
                                      std::string& error) {
    const int64_t tokens = (int64_t) c.ids.size(), j = tokens / kChunk;
    if (!c.imgs.empty() || !c.stage_parts.empty() || tokens < 1 || j >= (int64_t) keys.size())
        return fail(error, "a checkpoint with pictures or stage parts is not stored");
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    std::vector<ConversationKvLane> lanes;
    if (!conversation_kv_lanes(ss, g, draft, lanes, error)) return false;
    const int64_t tail = tokens - j * kChunk, cells = tail_cells(lanes, tail, page_of(lanes, g));
    if (chunk_bytes(lanes, cells) > opt_.staging_bytes) return fail(error, "staging buffer too small for a tail");
    const auto at = regions(lanes, staging_, cells);
    if (cells && !lanes_copy(lanes, g, j * kChunk, cells, at, false, error)) return false;
    std::vector<kvstore::Piece> pieces = {{c.gdn.data(), c.gdn.size()}, {c.ple.data(), c.ple.size()},
                                          {c.tails.data(), c.tails.size()}, {c.dead.data(), c.dead.size()},
                                          {c.block_pos.data(), c.block_pos.size()}};
    for (size_t i = 0; i < lanes.size(); ++i) pieces.push_back({at[i], lanes[i].bytes(cells)});
    const std::vector<int32_t> tail_ids(c.ids.begin() + j * kChunk, c.ids.end());
    if (!store_.put_snapshot(keys[(size_t) j], tokens, tail_ids, c.branch ? kvstore::kFlagBranch : 0, pieces, error))
        return false;
    ++c_.snapshot_writes;
    return true;
}

bool ConversationTier::after_request(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
                                     const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks,
                                     int64_t rewrite_from, std::optional<uint64_t> available, std::string& error) {
    last_request_ = Clock::now();
    if (!on() || checks.empty() || (int64_t) ids.size() < opt_.min_tokens) return true;
    for (const auto& c : checks)
        if (!c.imgs.empty() || !c.stage_parts.empty()) return true;   // text, one GPU: pictures are not stored
    const auto t0 = Clock::now();
    if (!ensure_staging(error) || !device_idle(error)) return false;
    const auto keys = kvstore::chain_keys(seed(cvec), ids.data(), ids.size());
    int64_t deepest = 0;
    for (const auto& c : checks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    if (!write_chunks(ss, g, draft, keys, ids, deepest, std::max<int64_t>(rewrite_from, 0), error)) return false;
    std::vector<const ConversationCheckpoint*> chain;
    for (const auto& c : checks)
        if (c.ids.size() <= ids.size() && std::equal(c.ids.begin(), c.ids.end(), ids.begin())) chain.push_back(&c);
    std::sort(chain.begin(), chain.end(), [](auto* a, auto* b) { return a->ids.size() < b->ids.size(); });
    int64_t covered = 0;   // the deepest point stored or waiting so far on this path
    for (const ConversationCheckpoint* c : chain) {
        const int64_t L = (int64_t) c->ids.size();
        const int64_t stored = std::max(stored_on_path(keys, ids, L), covered);
        if (stored == L || std::find(pending_.begin(), pending_.end(), L) != pending_.end()) { covered = L; continue; }
        if (!tier_snapshot_worth(L, stored, c->branch || c->user_turn || c->pinned, opt_.min_gain)) {
            ++c_.snapshot_skipped;
            continue;
        }
        const uint64_t bytes = c->bytes();
        if (tier_write_now(available, opt_.floor_bytes, bytes)) {
            if (!write_snapshot(ss, g, draft, keys, *c, error)) return false;
            ++c_.snapshot_now;
        } else {
            pending_.push_back(L);
            ++c_.snapshot_deferred;
        }
        covered = L;
    }
    ++c_.persists;
    c_.persist_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return store_.sync(error);
}

bool ConversationTier::flush(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
                             const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks,
                             std::string& error, int64_t above) {
    if (!on() || !waiting_above(above)) return true;
    if (!ensure_staging(error) || !device_idle(error)) return false;
    const auto keys = kvstore::chain_keys(seed(cvec), ids.data(), ids.size());
    std::vector<int64_t> keep;
    for (const int64_t L : pending_) {
        if (L <= above) { keep.push_back(L); continue; }
        const ConversationCheckpoint* found = nullptr;
        for (const auto& c : checks)
            if ((int64_t) c.ids.size() == L && c.ids.size() <= ids.size() &&
                std::equal(c.ids.begin(), c.ids.end(), ids.begin())) found = &c;
        if (!found) { ++c_.snapshot_lost; continue; }
        if (!write_snapshot(ss, g, draft, keys, *found, error)) { pending_ = keep; return false; }
    }
    pending_ = keep;
    return store_.sync(error);
}

bool ConversationTier::spill(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
                             const std::vector<int32_t>& ids, const ConversationCheckpoint& c, std::string& error) {
    const auto it = std::find(pending_.begin(), pending_.end(), (int64_t) c.ids.size());
    if (!on() || it == pending_.end()) return true;
    pending_.erase(it);
    if (c.ids.size() > ids.size() || !std::equal(c.ids.begin(), c.ids.end(), ids.begin())) { ++c_.snapshot_lost; return true; }
    if (!ensure_staging(error) || !device_idle(error)) return false;
    const auto keys = kvstore::chain_keys(seed(cvec), ids.data(), ids.size());
    if (!write_snapshot(ss, g, draft, keys, c, error)) return false;
    ++c_.snapshot_spilled;
    return true;
}

bool ConversationTier::idle_due() const {
    return !pending_.empty() && Clock::now() - last_request_ >= std::chrono::seconds(std::max<int64_t>(opt_.idle_s, 0));
}

ConversationTier::Hit ConversationTier::best(const std::vector<int32_t>& prompt, bool cvec) const {
    Hit h;
    if (!on() || prompt.size() < 2) return h;
    const auto keys = kvstore::chain_keys(seed(cvec), prompt.data(), prompt.size());
    if (const kvstore::Record* r = store_.best_snapshot(keys, prompt.data(), prompt.size())) {
        h.tokens = r->tokens;
        h.cvec = cvec;
    }
    return h;
}

bool ConversationTier::restore(const Hit& hit, const SessionState& ss, const ModelGeometry& g, const QsaState* draft,
                               const std::vector<int32_t>& prompt, ConversationCheckpoint& out, std::string& error) {
    const auto t0 = Clock::now();
    if (!on() || hit.tokens <= 0) return fail(error, "no snapshot");
    ++c_.restore_failures;   // undone at the end
    if (!ensure_staging(error) || !device_idle(error)) return false;
    const auto keys = kvstore::chain_keys(seed(hit.cvec), prompt.data(), prompt.size());
    const kvstore::Record* snap = store_.best_snapshot(keys, prompt.data(), prompt.size());
    if (!snap || snap->tokens != hit.tokens) return fail(error, "the snapshot is gone");
    const int64_t tokens = snap->tokens, whole = tokens / kChunk;
    std::vector<ConversationKvLane> lanes;
    if (!conversation_kv_lanes(ss, g, draft, lanes, error)) return false;
    const int64_t cells = tail_cells(lanes, tokens - whole * kChunk, page_of(lanes, g));
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    const size_t layers = (size_t) std::max<int64_t>(ss.qsa_alloc, 0);
    out = ConversationCheckpoint{};
    out.gdn.resize(z.gdn); out.ple.resize(ss.ple_hist ? z.ple : 0);
    out.tails.resize(layers * z.tail); out.dead.resize(layers * z.dead); out.block_pos.resize(layers * z.block_pos);
    const size_t state = out.gdn.size() + out.ple.size() + out.tails.size() + out.dead.size() + out.block_pos.size();
    // everything is checked before the first pool write: a snapshot of another layout is refused here
    if (snap->payload != state + chunk_bytes(lanes, cells) || chunk_bytes(lanes, cells) > opt_.staging_bytes)
        return fail(error, "snapshot payload does not fit this session");
    const size_t per_chunk = chunk_bytes(lanes, kChunk);
    const int64_t batch = std::max<int64_t>(1, (int64_t) (opt_.staging_bytes / std::max<size_t>(per_chunk, 1)));
    for (int64_t j0 = 0; j0 < whole; j0 += batch) {
        const int64_t m = std::min(batch, whole - j0);
        const auto at = regions(lanes, staging_, m * kChunk);
        for (int64_t k = 0; k < m; ++k) {
            const int64_t j = j0 + k;
            const kvstore::Record* r = store_.chunk(keys[(size_t) j + 1]);
            if (!r || r->payload != per_chunk) return fail(error, "a chunk is missing or of another layout");
            std::vector<kvstore::Store::Out> dst;
            for (size_t i = 0; i < lanes.size(); ++i)
                dst.push_back({at[i] + (size_t) k * lanes[i].bytes(kChunk), lanes[i].bytes(kChunk)});
            if (!store_.read(*r, dst, error)) return false;
            const int64_t next = (j + 1) * kChunk;
            if (r->next_token >= 0 && next < (int64_t) prompt.size() && prompt[(size_t) next] != r->next_token)
                ++c_.draft_lookahead_miss;   // the draft layer's last cell of this chunk was made for another token
        }
        if (!lanes_copy(lanes, g, j0 * kChunk, m * kChunk, at, true, error)) return false;
    }
    const auto at = regions(lanes, staging_, cells);
    std::vector<kvstore::Store::Out> dst = {{out.gdn.data(), out.gdn.size()}, {out.ple.data(), out.ple.size()},
                                            {out.tails.data(), out.tails.size()}, {out.dead.data(), out.dead.size()},
                                            {out.block_pos.data(), out.block_pos.size()}};
    for (size_t i = 0; i < lanes.size(); ++i) dst.push_back({at[i], lanes[i].bytes(cells)});
    if (!store_.read(*snap, dst, error)) return false;
    if (cells && !lanes_copy(lanes, g, whole * kChunk, cells, at, true, error)) return false;
    std::vector<const QsaState*> states;
    for (const auto& l : lanes)
        if (std::find(states.begin(), states.end(), l.state) == states.end()) states.push_back(l.state);
    for (const QsaState* st : states)
        if (!conversation_kv_refresh(*st, g, tokens, error)) return false;
    out.ids.assign(prompt.begin(), prompt.begin() + tokens);
    out.branch = (snap->flags & kvstore::kFlagBranch) != 0;
    if (!conversation_checkpoint_validate(out, ss, g, error)) return false;
    --c_.restore_failures;
    ++c_.restores;
    c_.restored_tokens += (uint64_t) tokens;
    c_.restore_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return true;
}

std::string ConversationTier::stats_line() const {
    const auto& s = store_.stats();
    char b[1024];
    std::snprintf(b, sizeof b,
                  "kv store: records=%llu chunks=%llu snapshots=%llu disk_bytes=%llu segments=%llu written_bytes=%llu "
                  "chunk_writes=%llu chunk_dedup=%llu snapshot_writes=%llu snapshot_now=%llu snapshot_deferred=%llu "
                  "snapshot_spilled=%llu snapshot_skipped=%llu snapshot_lost=%llu waiting=%zu restores=%llu "
                  "restored_tokens=%llu restore_failures=%llu restore_ms=%.1f persist_ms=%.1f rewrite_same=%llu "
                  "rewrite_differ=%llu draft_lookahead_miss=%llu read_errors=%llu gc_segments=%llu gc_copied=%llu "
                  "gc_dropped=%llu torn=%llu",
                  (unsigned long long) s.records, (unsigned long long) s.chunks, (unsigned long long) s.snapshots,
                  (unsigned long long) s.disk_bytes, (unsigned long long) s.segments, (unsigned long long) s.written_bytes,
                  (unsigned long long) c_.chunk_writes, (unsigned long long) c_.chunk_dedup,
                  (unsigned long long) c_.snapshot_writes, (unsigned long long) c_.snapshot_now,
                  (unsigned long long) c_.snapshot_deferred, (unsigned long long) c_.snapshot_spilled,
                  (unsigned long long) c_.snapshot_skipped, (unsigned long long) c_.snapshot_lost, pending_.size(),
                  (unsigned long long) c_.restores, (unsigned long long) c_.restored_tokens,
                  (unsigned long long) c_.restore_failures, c_.restore_ms, c_.persist_ms,
                  (unsigned long long) c_.rewrite_same, (unsigned long long) c_.rewrite_differ,
                  (unsigned long long) c_.draft_lookahead_miss, (unsigned long long) s.read_errors,
                  (unsigned long long) s.gc_segments, (unsigned long long) s.gc_copied, (unsigned long long) s.gc_dropped,
                  (unsigned long long) s.torn);
    return b;
}

}  // namespace strata::core
