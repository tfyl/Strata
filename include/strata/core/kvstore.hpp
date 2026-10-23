// tfyl: the conversation cache's disk tier - a content-addressed store of K/V chunks and running-state snapshots.
//
// A conversation is a chain of 256-token CHUNKS, each keyed by a chained hash: the key after j chunks hashes the key
// after j - 1 with the chunk's token ids, starting from a seed that fingerprints everything the bytes depend on (the
// model and its files, the engine configuration, the K/V layout, the steering mode).  So a shared prefix is stored
// once across turns, branches and conversations, and any chat is rebuilt by walking its chain.  A chunk's payload
// is its cells' K/V of every attention layer and the draft layer.  A SNAPSHOT is a running state at any token count:
// it hangs off the key of the chunks below it with the tokens past them (up to 255), and carries those tail cells'
// K/V too.  A restore walks the prompt's chunk keys while they are stored and takes the deepest snapshot on the way.
//
// On disk: append-only segment files (seg-NNNNNNNN.kvs, `segment_bytes` each at most) of records, each a 4 KiB header
// page and a payload padded to 4 KiB, written with O_DIRECT where the filesystem takes it (no page cache: RAM is
// scarce).  Every header and payload carries a digest, so a torn write at the end of a segment (a crash) is found
// when the store opens and dropped; nothing is fsynced per record, only when a segment is sealed and on sync().
// The index lives in RAM (~150 bytes a record) and is rebuilt by scanning the record headers at open.  Over its
// byte budget the store deletes its oldest segment, copying forward the records read or matched since they were
// written (once: a second chance), so a segment is never compacted and dropping is the default - this is a cache.
//
// The store knows nothing of CUDA or the model: payloads are bytes, laid out by the caller (generate.cpp).  It is
// used from one thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core::kvstore {

inline constexpr int64_t kChunkTokens = 256;
inline constexpr size_t kPage = 4096;
inline constexpr size_t kMaxTail = (size_t) kChunkTokens - 1;

struct Key {
    uint64_t a = 0, b = 0;
    bool operator==(const Key& o) const { return a == o.a && b == o.b; }
    bool operator!=(const Key& o) const { return !(*this == o); }
    bool empty() const { return a == 0 && b == 0; }
};
struct KeyHash {
    size_t operator()(const Key& k) const { return (size_t) (k.a ^ (k.b * 0x9E3779B97F4A7C15ull)); }
};

/// The chain's seed: every value the cached bytes depend on, hashed (a different value never matches).
Key chain_seed(const std::vector<uint64_t>& identity);
/// The key after one more chunk of `n` token ids.
Key chain_next(const Key& prev, const int32_t* tokens, size_t n);
/// keys[j] = the key after j whole chunks of `tokens` (keys[0] = seed), for j up to n / kChunkTokens.
std::vector<Key> chain_keys(const Key& seed, const int32_t* tokens, size_t n);
/// A payload handed in pieces (the caller's buffers, any alignment).
struct Piece {
    const void* data = nullptr;
    size_t n = 0;
};
/// A payload's digest, as the store records it (Record::sum).
uint64_t digest(const std::vector<Piece>& payload);

enum class Kind : uint32_t { chunk = 1, snapshot = 2 };

struct Record {
    Kind kind = Kind::chunk;
    uint32_t seg = 0;            // segment number
    uint64_t off = 0;            // the header page's offset in it
    uint64_t payload = 0;        // payload bytes (unpadded)
    uint64_t sum = 0;            // payload digest
    uint64_t seq = 0;            // write order, store-wide
    uint64_t used = 0;           // the store's clock when last read or matched (0: never since written)
    Key key, parent;             // chunk: its key and the one below; snapshot: the key of the chunks below it
    int32_t next_token = -1;     // chunk: the token after it (the draft layer's last cell depends on it)
    int64_t tokens = 0;          // snapshot: its token count
    uint32_t flags = 0;          // snapshot: kFlagBranch, ...
    std::vector<int32_t> tail;   // snapshot: the tokens past its whole chunks
    size_t bytes_on_disk() const { return kPage + (payload + kPage - 1) / kPage * kPage; }
};
inline constexpr uint32_t kFlagBranch = 1;

struct Stats {
    uint64_t records = 0, chunks = 0, snapshots = 0, disk_bytes = 0, segments = 0;
    uint64_t written_bytes = 0, chunks_written = 0, snapshots_written = 0;   // since open
    uint64_t gc_segments = 0, gc_dropped = 0, gc_copied = 0, read_errors = 0, torn = 0;
};

class Store {
public:
    struct Options {
        std::string dir;
        uint64_t max_bytes = 100ull << 30;     // the budget: over it, the oldest segment goes
        uint64_t segment_bytes = 256ull << 20;
        bool direct = true;                    // O_DIRECT when the filesystem takes it
    };
    Store();
    ~Store();
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    /// Opens (creates) the directory, scans its segments into the index (a torn or foreign tail is dropped) and
    /// keeps to the budget.  False with `error` when the directory cannot be used.
    bool open(const Options& o, std::string& error);
    bool is_open() const { return fd_ >= 0; }
    const Options& options() const { return opt_; }

    const Record* chunk(const Key& k) const;
    /// Snapshots that hang off key `at` (whole chunks below them), any tail.
    std::vector<const Record*> snapshots_at(const Key& at) const;
    /// The deepest snapshot a prompt can resume from: `keys` = chain_keys of the prompt, `n` its tokens.  Its whole
    /// chunks must all be stored, its tail must be the prompt's next tokens and it must end before the prompt's
    /// last token (which starts the next verify window).  nullptr: none.
    const Record* best_snapshot(const std::vector<Key>& keys, const int32_t* prompt, size_t n) const;
    /// How many leading chunks of `keys` are stored (keys[1..j] all present).
    size_t stored_chunks(const std::vector<Key>& keys) const;

    /// A chunk, first writer wins: an existing key is left as it is (true, `*written` false).
    bool put_chunk(const Key& k, const Key& parent, int32_t next_token, const std::vector<Piece>& payload,
                   std::string& error, bool* written = nullptr);
    /// A snapshot of `tokens` tokens hanging off `at` with `tail` past the whole chunks.  One at the same place
    /// (key, tail) is replaced.
    bool put_snapshot(const Key& at, int64_t tokens, const std::vector<int32_t>& tail, uint32_t flags,
                      const std::vector<Piece>& payload, std::string& error);
    /// Reads a record's payload into `dst` (`n` = its payload bytes) and checks its digest; marks it used.
    bool read(const Record& r, void* dst, size_t n, std::string& error);
    /// The same, scattered into pieces whose sizes add up to the payload.
    struct Out { void* data = nullptr; size_t n = 0; };
    bool read(const Record& r, const std::vector<Out>& dst, std::string& error);
    /// Marks records used (a match that is not read: a chunk already in the pools).
    void touch(const Record& r);
    /// Flushes the active segment's data (fdatasync).
    bool sync(std::string& error);

    const Stats& stats() const { return st_; }

private:
    struct Index;
    bool append(Record& r, const std::vector<Piece>& payload, std::string& error);
    bool roll(std::string& error);
    bool keep_budget(std::string& error);
    bool scan(uint32_t seg, std::string& error);
    void index_put(Record&& r);
    void index_drop(const Record& r);
    std::string seg_path(uint32_t seg) const;

    Options opt_;
    int fd_ = -1;                 // the active segment
    uint32_t seg_ = 0;            // its number
    uint64_t end_ = 0;            // its append offset
    bool direct_ = false;
    uint64_t seq_ = 0, clock_ = 0;
    std::unique_ptr<Index> ix_;
    uint8_t* bounce_ = nullptr;   // aligned staging for O_DIRECT
    size_t bounce_n_ = 0;
    Stats st_;
};

}  // namespace strata::core::kvstore
