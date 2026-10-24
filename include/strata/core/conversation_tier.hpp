// tfyl: the conversation cache's disk tier as the serve loop uses it (`--kv-store DIR`).  kvstore.hpp is the store;
// this lays a session's K/V and running state out as its records and decides what goes to disk and when.
//
//   * After a request, every whole 256-token chunk of the conversation below its deepest checkpoint that the store
//     does not hold yet is written (the chunks are final: the next request of any conversation that shares them
//     reads the same ids).  Chunks the request rewrote that the store already holds are compared, not written
//     (the first writer stays): `rewrite_same` / `rewrite_differ` count how often the bytes changed.
//   * A checkpoint is worth a disk snapshot (~118 MB) when it is `min_gain` tokens past the deepest one stored on
//     its path, or a kept point (the system-prompt root, a branch point, the newest user message).  It is written
//     right away when RAM is short (under the floor plus two snapshots); else it waits in RAM, where the next turn
//     of the same conversation resumes from it anyway, and goes to disk when the conversation leaves the session
//     (another one's request), when retention drops it, after `idle_s` without requests, or at QUIT.  A crash loses
//     only the snapshots still waiting.
//   * A request that matches nothing in RAM as deep as the store's deepest snapshot on its path restores from disk:
//     the chunks and the snapshot's tail cells into the K/V pools, its running state as the checkpoint to mount.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/kvstore.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata::core {
struct SessionState;
struct ModelGeometry;
struct QsaState;

/// Whether a checkpoint of `tokens` tokens is worth a disk snapshot: past the deepest one stored on its path
/// (`stored`, 0: none), by `min_gain` tokens unless it is a kept point.
inline bool tier_snapshot_worth(int64_t tokens, int64_t stored, bool kept, int64_t min_gain) {
    return tokens > stored && (kept || tokens - stored >= min_gain);
}
/// Write it now (true) or let it wait in RAM: now when available RAM is unknown or under the floor plus two
/// snapshots - the engine may have to drop RAM checkpoints soon, and then only the disk copy is left.
inline bool tier_write_now(std::optional<uint64_t> available, uint64_t floor, uint64_t snapshot_bytes) {
    return !available || *available < floor + 2 * snapshot_bytes;
}
/// tfyl.6: whether a request makes the session leave its live conversation, so that conversation's tip is kept
/// (subject to the value gate) before the pools are rewritten: another conversation comes in (a restore, a borrow,
/// a batch slot), or the prompt leaves the live tokens before their end - a new chat read from token 0 or from a
/// shared root, a side request that forks off further back.  A prompt that only re-renders the last turn leaves too;
/// its tip is then a few hundred tokens past the turn header and the value gate drops it.
inline bool tier_leaves(bool switching, bool live_ok, int64_t live_tokens, int64_t shared_with_live) {
    return live_ok && live_tokens > 0 && (switching || shared_with_live < live_tokens);
}
/// Stage 3: how many bytes the RAM snapshot pool may hold - what is available above the floor and a headroom, the
/// pool's own bytes counted as its own (they come back when it shrinks), at least `min_entries` snapshots (the
/// live resume point and the newest user message of the conversation that just left).  Unknown RAM: the minimum.
inline uint64_t tier_pool_capacity(std::optional<uint64_t> available, uint64_t pool_bytes, uint64_t floor,
                                   uint64_t headroom, uint64_t snapshot_bytes, uint64_t min_entries) {
    const uint64_t least = min_entries * snapshot_bytes;
    if (!available) return least;
    const uint64_t have = *available + pool_bytes, keep = floor + headroom;
    return have > keep + least ? have - keep : least;
}
/// Marconi's utility (FLOP-aware eviction, prefix caching for hybrid models): recency plus alpha times the compute a
/// snapshot saves per byte, both normalised to [0, 1] over the candidates; a kept point (a branch point, the newest
/// user message) is worth one more.  The lowest goes first.  Every snapshot here is the same size, so the saving per
/// byte is the tokens it saves over the next snapshot on its path.
inline double tier_utility(double recency01, double saved01, bool kept, double alpha = 1.0) {
    return recency01 + alpha * saved01 + (kept ? 1.0 : 0.0);
}

class ConversationTier {
public:
    struct Options {
        std::string dir;
        uint64_t max_bytes = 100ull << 30;
        int64_t min_tokens = 0;        // conversations shorter than this are not stored
        int64_t min_gain = 4096;       // tokens a snapshot must add over the stored one on its path
        int64_t idle_s = 300;          // waiting snapshots go to disk after this long without requests
        uint64_t floor_bytes = 2ull << 30;   // the RAM floor (--conversation-cache-min-free-mib)
        size_t staging_bytes = 32u << 20;    // the copy buffer between the pools and the store
        // stage 3: the RAM snapshot pool (conversations that left the session keep their resume states here)
        bool pool = false;
        uint64_t pool_headroom = 512ull << 20;   // RAM left over the floor before the pool takes any
        uint64_t pool_min = 2;                   // snapshots it keeps whatever RAM says
        uint64_t pool_max = 1536ull << 20;       // and at most this, whatever RAM says (--kv-store-pool-mib)
        double alpha = 1.0;                      // Marconi's weight of the compute saved against recency
        // tfyl.7: what the store may write in a rolling 24 h (chunks, snapshots and GC copies; 0 = no limit).
        // Over it, a conversation is stored only past a raised threshold (min_tokens doubled per quarter of the
        // budget over) unless it is mostly stored already, and only fork points (a branch, the newest user message)
        // still go to disk as snapshots
        uint64_t budget_bytes_day = 0;
    };
    /// What best() found (restore() looks it up again: a store write in between may move the record).
    struct Hit {
        int64_t tokens = 0;
        bool cvec = false;
        bool pooled = false;   // its running state is in the RAM pool (no disk read for it)
    };
    struct Counters {
        uint64_t persists = 0, chunk_writes = 0, chunk_dedup = 0, snapshot_writes = 0, snapshot_skipped = 0;
        uint64_t snapshot_now = 0, snapshot_deferred = 0, snapshot_spilled = 0, snapshot_lost = 0;
        uint64_t restores = 0, restored_tokens = 0, restore_failures = 0, draft_lookahead_miss = 0;
        uint64_t rewrite_same = 0, rewrite_differ = 0, lineage_skipped = 0, tip_taken = 0, tip_skipped = 0;
        uint64_t pool_adopted = 0, pool_restores = 0, pool_spilled = 0, pool_dropped = 0;
        // tfyl.7: states not written because a deeper state of the same conversation is in RAM or on disk, or the
        // budget was spent; requests whose conversation was not stored because of the budget's raised threshold
        uint64_t snapshot_superseded = 0, snapshot_budget_dropped = 0, budget_skipped = 0;
        double restore_ms = 0, persist_ms = 0;
    };

    ConversationTier() = default;
    ConversationTier(const ConversationTier&) = delete;
    ConversationTier& operator=(const ConversationTier&) = delete;
    ~ConversationTier();

    /// `identity`: what the bytes depend on (the session file's model and config fingerprints, the geometry, the
    /// K/V layout); the steering mode is added per call.
    bool open(const Options& o, const std::vector<uint64_t>& identity, std::string& error);
    bool on() const { return store_.is_open(); }
    const Options& options() const { return opt_; }

    /// After a request: write the chunks below the deepest checkpoint, compare the ones it rewrote (cells from
    /// `rewrite_from`), queue or write the snapshots worth it.  `ids` = the session's tokens (its pools hold them),
    /// `checks` its checkpoint chain, `available` the RAM available now.  The device is idle.
    bool after_request(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
                       const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks,
                       int64_t rewrite_from, std::optional<uint64_t> available, std::string& error);
    /// The waiting snapshots that are still in `checks` go to disk (the session leaves this conversation, retention
    /// is dropping them, it was idle, QUIT).  The pools must still hold `ids`.
    /// `above`: only the ones longer than that (a request that continues this conversation from `above` rewrites the
    /// cells past it; the others stay valid and keep waiting).
    bool flush(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
               const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks, std::string& error,
               int64_t above = 0);
    /// Whether a snapshot longer than `above` is waiting.
    bool waiting_above(int64_t above) const {
        for (int64_t L : pending_) if (L > above) return true;
        return false;
    }
    /// One checkpoint retention is about to drop: written if it was waiting.
    bool spill(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
               const std::vector<int32_t>& ids, const ConversationCheckpoint& c, std::string& error);
    bool waiting() const { return !pending_.empty(); }
    bool idle_due() const;
    /// Stage 3 (Options::pool): the session leaves this conversation (or rewrites past `above`): the waiting
    /// snapshots longer than `above` MOVE into the RAM pool with their tail cells' K/V - no disk write - and leave
    /// `checks`.  Then the pool keeps to its capacity (trim).  Without the pool this is flush().
    bool adopt(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
               const std::vector<int32_t>& ids, std::vector<ConversationCheckpoint>& checks, int64_t above,
               std::optional<uint64_t> available, std::string& error);
    /// The session leaves this conversation (tier_leaves): `live_state` (the running state at the tip, ids = `ids`)
    /// joins the waiting snapshots as a kept point when it is min_gain past the deepest point on its path (stored,
    /// waiting or pooled), with the K/V up to the tip stored (the chunks past the deepest checkpoint compared, so the
    /// tip's lineage is the bytes it was taken over), and everything waiting longer than `above` moves into the pool
    /// (adopt; `above`: where this request resumes, 0 for another conversation).  So a conversation that comes back
    /// resumes where it left - an agent that echoes its reply continues from the tip - without a parked K/V image.
    /// Whether this conversation's tip (ids.size()) is worth a state when the session leaves it - checked before the
    /// caller copies the running state off the device.
    bool tip_worth(const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks, bool cvec) const;
    bool leave(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
               const std::vector<int32_t>& ids, std::vector<ConversationCheckpoint>& checks,
               ConversationCheckpoint&& live_state, int64_t above, std::optional<uint64_t> available,
               std::string& error);
    /// The pool to its capacity for `available` RAM: the lowest Marconi utility goes first, to disk when it is not
    /// stored yet (spilled), else just dropped.
    bool trim(std::optional<uint64_t> available, std::string& error);
    /// Every snapshot only RAM holds goes to disk (QUIT, FLUSH; IDLE for the ones past idle_s).
    bool save_pool(bool only_idle, std::string& error);
    /// Whether a conversation of `tokens` tokens is the tier's to keep (the RAM conversation cache need not park it).
    bool covers(int64_t tokens) const { return on() && opt_.pool && tokens >= opt_.min_tokens; }
    size_t pool_entries() const { return pool_.size(); }
    uint64_t pool_bytes() const;
    /// The waiting list is no longer this session's (the pools were rewritten, or #879 dropped everything).
    void forget() { c_.snapshot_lost += pending_.size(); pending_.clear(); }

    /// The deepest stored snapshot `prompt` can resume from (nullptr when none).
    Hit best(const std::vector<int32_t>& prompt, bool cvec) const;
    /// The hit's chunks and tail cells into the pools (refreshed), its running state into `out` (its ids from
    /// `prompt`).  False: nothing usable, and the pools may be partly written - the caller reads from token 0.
    bool restore(const Hit& hit, const SessionState& ss, const ModelGeometry& g, const QsaState* draft,
                 const std::vector<int32_t>& prompt, ConversationCheckpoint& out, std::string& error);

    const Counters& counters() const { return c_; }
    /// tfyl.7: the store's writes in the rolling 24 h, whether that is over the budget, and the conversation length a
    /// new conversation needs to be stored now (min_tokens, raised over the budget).
    uint64_t spent_24h() const;
    bool over_budget() const;
    int64_t admit_min_tokens() const;
    const kvstore::Stats& store_stats() const { return store_.stats(); }
    /// One line for the log and the server's /metrics ("kv store: k=v ...").
    std::string stats_line() const;

private:
    bool write_chunks(const SessionState& ss, const ModelGeometry& g, const QsaState* draft,
                      const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& ids, int64_t upto,
                      int64_t rewrite_from, std::string& error);
    bool write_snapshot(const SessionState& ss, const ModelGeometry& g, const QsaState* draft,
                        const std::vector<kvstore::Key>& keys, const ConversationCheckpoint& c, std::string& error);
    struct PoolEntry {
        kvstore::Key at;                    // the key of the whole chunks below it
        bool cvec = false;
        std::vector<int32_t> tail;          // the tokens past them
        ConversationCheckpoint state;       // its running state (ids empty: they are the prompt's)
        std::vector<uint8_t> tail_kv;       // the tail cells' K/V, lane after lane
        int64_t tokens = 0, saved = 0;      // its length; the tokens it saves over the next point on its path
        uint64_t used = 0;
        bool kept = false, on_disk = false;
        bool tip = false;                   // the conversation's live tip when it left (not a fork point)
        uint64_t group = 0;                 // the adopt it came in with: one conversation's states share it
        uint64_t lineage = 0;               // the chunks below it when it was taken (kvstore Record::lineage)
        std::chrono::steady_clock::time_point since{};
        size_t bytes() const { return state.bytes() + tail_kv.capacity() + tail.capacity() * 4; }
    };
    bool put_entry(const PoolEntry& e, std::string& error);
    /// tfyl.7: whether a state should reach disk at all: not when a deeper state of its own conversation is in the
    /// pool or was written (a fork point - a branch, the newest user message - is exempt: the main line comes back to
    /// it), and over the budget only a fork point that saves 2 x min_gain.  Counts what it drops.
    bool disk_worthy(const PoolEntry& e, bool budget = true);
    /// Whether a key `depth_a` chunks deep is on the path of another `depth_b` deep (its parents walked back).
    bool on_path(const kvstore::Key& a, size_t depth_a, const kvstore::Key& b, size_t depth_b) const;

    /// Whether this conversation is stored: past admit_min_tokens(), or mostly stored already.
    bool admitted(const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& ids) const;
    const PoolEntry* pool_best(const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& prompt, bool cvec,
                               const std::vector<uint64_t>& lin) const;
    int64_t stored_on_path(const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& ids, int64_t tokens) const;
    /// lin[j] = the lineage of the stored chunks keys[1..j], for as long as they are all stored.
    std::vector<uint64_t> lineages(const std::vector<kvstore::Key>& keys) const;
    /// The deepest stored snapshot on the path whose lineage is the stored chunks' (nullptr: none).
    const kvstore::Record* valid_snapshot(const std::vector<kvstore::Key>& keys, const std::vector<uint64_t>& lin,
                                          const int32_t* prompt, size_t n) const;
    kvstore::Key seed(bool cvec) const;
    bool ensure_staging(std::string& error);

    Options opt_;
    std::vector<uint64_t> identity_;
    kvstore::Store store_;
    uint8_t* staging_ = nullptr;
    std::vector<int64_t> pending_;   // checkpoint lengths waiting in RAM
    std::vector<PoolEntry> pool_;    // stage 3: states of conversations that left the session
    uint64_t clock_ = 0, group_ = 0;
    mutable std::vector<std::pair<std::chrono::steady_clock::time_point, uint64_t>> spend_;   // (when, bytes)
    mutable uint64_t spend_seen_ = 0;                                                          // written_bytes accounted
    std::vector<std::pair<uint64_t, int64_t>> group_disk_;                             // group -> deepest written
    std::chrono::steady_clock::time_point last_request_{};
    mutable Counters c_;   // (lineage_skipped counts in the const lookups)
};

}  // namespace strata::core
