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
        uint64_t rewrite_same = 0, rewrite_differ = 0;
        uint64_t pool_adopted = 0, pool_restores = 0, pool_spilled = 0, pool_dropped = 0;
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
    /// The session leaves this conversation for another one: its K/V up to the live tip is stored (whole chunks),
    /// `live_state` (the running state at the tip, ids = `ids`) joins the waiting snapshots as a kept point, and
    /// everything waiting moves into the pool (adopt).  So a conversation that comes back resumes exactly where it
    /// left - an agent that echoes its reply continues from the tip - without a RAM-parked K/V image.
    bool leave(const SessionState& ss, const ModelGeometry& g, const QsaState* draft, bool cvec,
               const std::vector<int32_t>& ids, std::vector<ConversationCheckpoint>& checks,
               ConversationCheckpoint&& live_state, std::optional<uint64_t> available, std::string& error);
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
        std::chrono::steady_clock::time_point since{};
        size_t bytes() const { return state.bytes() + tail_kv.capacity() + tail.capacity() * 4; }
    };
    bool put_entry(const PoolEntry& e, std::string& error);
    const PoolEntry* pool_best(const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& prompt, bool cvec,
                               size_t whole) const;
    int64_t stored_on_path(const std::vector<kvstore::Key>& keys, const std::vector<int32_t>& ids, int64_t tokens) const;
    kvstore::Key seed(bool cvec) const;
    bool ensure_staging(std::string& error);

    Options opt_;
    std::vector<uint64_t> identity_;
    kvstore::Store store_;
    uint8_t* staging_ = nullptr;
    std::vector<int64_t> pending_;   // checkpoint lengths waiting in RAM
    std::vector<PoolEntry> pool_;    // stage 3: states of conversations that left the session
    uint64_t clock_ = 0;
    std::chrono::steady_clock::time_point last_request_{};
    Counters c_;
};

}  // namespace strata::core
