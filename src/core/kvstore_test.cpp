// CPU only (a temporary directory): the conversation cache's disk tier - chained keys, records that survive a
// reopen, a torn tail and a corrupt payload found and dropped, first writer wins, the deepest snapshot a prompt can
// resume from, and the budget's oldest-segment-first collection with one copy forward for what was used.
#include "strata/core/kvstore.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace strata::core::kvstore;
static int checks_run = 0;
static void check(bool ok, const char* label) {
    ++checks_run;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
static std::vector<uint8_t> bytes(size_t n, uint8_t salt) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t) (i * 31 + salt);
    return v;
}
static std::vector<int32_t> prompt(size_t n, int32_t base) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + (int32_t) i;
    return v;
}
static std::string tmpdir() {
    char t[] = "/tmp/kvstore-test-XXXXXX";
    const char* d = mkdtemp(t);
    check(d != nullptr, "temporary directory");
    return std::string(d) + "/store";
}
static void rmrf(const std::string& d) { std::string c = "rm -rf '" + d.substr(0, d.rfind('/')) + "'"; (void) !std::system(c.c_str()); }

int main() {
    const Key seed = chain_seed({1, 2, 3});
    // Keys: chained, deterministic, sensitive to every token, the seed and the order.
    {
        const auto p = prompt(1000, 7);
        const auto k = chain_keys(seed, p.data(), p.size());
        check(k.size() == 1 + 1000 / 256 && k[0] == seed, "one key per whole chunk, after the seed");
        check(chain_keys(seed, p.data(), p.size())[3] == k[3], "deterministic");
        auto q = p; q[300] += 1;
        const auto kq = chain_keys(seed, q.data(), q.size());
        check(kq[1] == k[1] && kq[2] != k[2] && kq[3] != k[3], "a change moves its chunk's key and every later one");
        check(chain_keys(chain_seed({1, 2, 4}), p.data(), p.size())[1] != k[1], "another identity, other keys");
        check(chain_seed({1, 2, 3}) == seed && !seed.empty(), "the seed");
    }
    const std::string dir = tmpdir();
    const auto p = prompt(2000, 100);
    const auto keys = chain_keys(seed, p.data(), p.size());   // 7 whole chunks
    const auto payload = bytes(3 * 4096 + 123, 5);            // not page-aligned: padded on disk
    const auto state = bytes(40000, 9);
    // Write, read back, reopen.
    {
        Store s;
        std::string err;
        check(s.open({dir, 1ull << 30, 1u << 20, true}, err), "open a new store");
        for (size_t j = 1; j <= 5; ++j) {
            bool w = false;
            check(s.put_chunk(keys[j], keys[j - 1], p[j * 256], {{payload.data(), 100}, {payload.data() + 100, payload.size() - 100}},
                              err, &w) && w, "a chunk in two pieces");
        }
        bool w = true;
        check(s.put_chunk(keys[3], keys[2], 0, {{state.data(), 10}}, err, &w) && !w, "first writer wins");
        std::vector<int32_t> tail(p.begin() + 1024, p.begin() + 1100);
        check(s.put_snapshot(keys[4], 1100, tail, 0, {{state.data(), state.size()}}, err), "a snapshot with a tail");
        check(s.put_snapshot(keys[2], 512, {}, kFlagBranch, {{state.data(), state.size()}}, err), "one at a chunk edge");
        check(s.stored_chunks(keys) == 5, "five chunks on the path");
        const Record* best = s.best_snapshot(keys, p.data(), p.size());
        check(best && best->tokens == 1100, "the deepest snapshot on the path");
        std::vector<uint8_t> got(state.size());
        check(best && s.read(*best, got.data(), got.size(), err) && got == state, "its payload back");
        const Record* c = s.chunk(keys[2]);
        std::vector<uint8_t> cg(payload.size());
        check(c && c->next_token == p[512] && s.read(*c, cg.data(), cg.size(), err) && cg == payload, "a chunk back");
        check(s.stats().chunks == 5 && s.stats().snapshots == 2 && s.stats().chunks_written == 5, "counts");
        check(s.sync(err), "sync");
    }
    {
        Store s;
        std::string err;
        check(s.open({dir, 1ull << 30, 1u << 20, true}, err), "reopen");
        check(s.stats().chunks == 5 && s.stats().snapshots == 2 && s.stats().torn == 0, "the index is rebuilt");
        const Record* best = s.best_snapshot(keys, p.data(), p.size());
        std::vector<uint8_t> got(state.size());
        check(best && best->tokens == 1100 && s.read(*best, got.data(), got.size(), err) && got == state, "and reads");
        // a prompt that leaves the path inside the snapshot's tail: the edge snapshot is the deepest it can use
        auto q = p; q[1050] = -5;
        const auto kq = chain_keys(seed, q.data(), q.size());
        best = s.best_snapshot(kq, q.data(), q.size());
        check(best && best->tokens == 512, "a diverging tail falls back");
        // a prompt that ends at the snapshot: the last prompt token starts the verify window, so not that one
        best = s.best_snapshot(keys, p.data(), 1100);
        check(best && best->tokens == 512, "not at the prompt's last token");
        // a missing chunk below a snapshot hides it
        auto r = p; r[300] = -1;
        const auto kr = chain_keys(seed, r.data(), r.size());
        best = s.best_snapshot(kr, r.data(), r.size());
        check(!best && s.stored_chunks(kr) == 1, "no snapshot past a chunk the store does not hold");
        check(!s.best_snapshot(chain_keys(chain_seed({9}), p.data(), p.size()), p.data(), p.size()), "other identity");
    }
    // A torn tail (a crash mid-write) and a corrupt payload.
    {
        const std::string seg = dir + "/seg-00000001.kvs";
        struct stat sb{};
        check(::stat(seg.c_str(), &sb) == 0, "the segment");
        const int fd = ::open(seg.c_str(), O_WRONLY);
        std::vector<uint8_t> junk(4096 * 3, 0xab);
        check(fd >= 0 && ::pwrite(fd, junk.data(), junk.size(), sb.st_size) == (ssize_t) junk.size(), "torn tail");
        // flip a byte inside the first chunk's payload
        const uint8_t x = 0x5a;
        check(::pwrite(fd, &x, 1, 4096 + 7) == 1, "corrupt a payload");
        ::close(fd);
        Store s;
        std::string err;
        check(s.open({dir, 1ull << 30, 1u << 20, true}, err), "reopen after a crash");
        check(s.stats().torn == 1 && s.stats().records == 7, "the torn tail is dropped, the records stay");
        std::vector<uint8_t> cg(payload.size());
        check(!s.read(*s.chunk(keys[1]), cg.data(), cg.size(), err) && s.stats().read_errors == 1, "digest mismatch");
        check(s.read(*s.chunk(keys[2]), cg.data(), cg.size(), err), "the others read");
        bool w = false;
        check(s.put_chunk(keys[6], keys[5], 0, {{payload.data(), payload.size()}}, err, &w) && w, "appends after it");
        Store t;
        check(t.open({dir, 1ull << 30, 1u << 20, true}, err) && t.stats().torn == 0 && t.chunk(keys[6]), "clean again");
    }
    rmrf(dir);
    // The budget: oldest segment first; what was used since it was written is copied forward once.
    {
        const std::string d2 = tmpdir();
        Store s;
        std::string err;
        const auto big = bytes(200000, 3);   // ~49 pages a record, 1 MiB segments: ~5 records a segment
        check(s.open({d2, 2u << 20, 1u << 20, true}, err), "a small budget");
        std::vector<Key> ks;
        Key prev = seed;
        for (int i = 0; i < 6; ++i) {
            int32_t t[256];
            for (int k = 0; k < 256; ++k) t[k] = i * 1000 + k;
            const Key k = chain_next(prev, t, 256);
            check(s.put_chunk(k, prev, 0, {{big.data(), big.size()}}, err), "fill");
            ks.push_back(k);
            prev = k;
        }
        std::vector<uint8_t> got(big.size());
        check(s.read(*s.chunk(ks[0]), got.data(), got.size(), err), "use the oldest");
        auto fill = [&](int i) {
            int32_t t[256];
            for (int k = 0; k < 256; ++k) t[k] = i * 1000 + k;
            const Key k = chain_next(prev, t, 256);
            check(s.put_chunk(k, prev, 0, {{big.data(), big.size()}}, err), "fill past the budget");
            ks.push_back(k);
            prev = k;
        };
        int i = 6;
        while (s.stats().gc_segments == 0) fill(i++);
        check(s.chunk(ks[0]) && s.stats().gc_copied == 1, "the used chunk was copied forward");
        check(!s.chunk(ks[1]) && s.stats().gc_dropped >= 1, "an unused old one was dropped");
        check(s.read(*s.chunk(ks[0]), got.data(), got.size(), err) && got == big, "its bytes intact");
        while (s.stats().gc_copied < 2 && i < 200) fill(i++);
        check(s.stats().gc_copied == 2 && s.chunk(ks[0]), "read again since its copy: copied again");
        while (s.chunk(ks[0]) && i < 400) fill(i++);
        check(!s.chunk(ks[0]) && s.stats().gc_copied == 2, "unread since: dropped, not copied forever");
        check(s.stats().disk_bytes <= (2u << 20) + (1u << 20) && s.stats().segments <= 4, "kept near the budget");
        Store t;
        check(t.open({d2, 2u << 20, 1u << 20, true}, err) && t.chunk(ks.back()) && !t.chunk(ks[0]) &&
              t.stats().chunks == s.stats().chunks, "the same records after a reopen");
        rmrf(d2);
    }
    // tfyl.7: the collection picks the segment with the least live data among the oldest, and copies forward only
    // what was used within copy_window_s; written bytes are counted by kind.
    {
        const std::string d4 = tmpdir();
        Store s;
        std::string err;
        Store::Options so{d4, (5u << 20) / 2, 1u << 20, true};   // 2.5 MiB: the 13th record crosses it
        so.copy_window_s = 0;   // nothing counts as recent: nothing is copied forward
        check(s.open(so, err), "a store for the collection order");
        const auto big = bytes(200000, 7);
        std::vector<Key> seg1, seg2;
        auto put = [&](int i, std::vector<Key>& into, bool replace = false, const Key* k0 = nullptr) {
            int32_t t[256];
            for (int k = 0; k < 256; ++k) t[k] = 777000 + i * 1000 + k;
            const Key k = k0 ? *k0 : chain_next(seed, t, 256);
            bool w = false;
            check(s.put_chunk(k, seed, 0, {{big.data(), big.size()}}, err, &w, replace) && w, "put");
            if (!k0) into.push_back(k);
        };
        for (int i = 0; i < 5; ++i) put(i, seg1);        // segment 1: five live chunks
        for (int i = 5; i < 10; ++i) put(i, seg2);       // segment 2: five chunks...
        for (const Key& k : seg2) put(0, seg2, true, &k);   // ...all replaced (segment 3): segment 2 is dead
        check(s.stats().gc_segments >= 1 && s.chunk(seg1[0]) && s.chunk(seg2[0]),
              "the dead segment 2 went first, not the older segment 1 with live data");
        check(s.stats().gc_copied == 0, "copy_window_s 0: nothing copied forward");
        check(s.stats().written_chunk_bytes == s.stats().written_bytes && s.stats().written_gc_bytes == 0 &&
              s.stats().written_snapshot_bytes == 0, "written bytes by kind");
        rmrf(d4);
    }
    // Buffered I/O (no O_DIRECT) works the same.
    {
        const std::string d3 = tmpdir();
        Store s;
        std::string err;
        check(s.open({d3, 1ull << 30, 1u << 20, false}, err), "buffered");
        check(s.put_chunk(keys[1], keys[0], 0, {{payload.data(), payload.size()}}, err), "write");
        std::vector<uint8_t> cg(payload.size());
        check(s.read(*s.chunk(keys[1]), cg.data(), cg.size(), err) && cg == payload, "read");
        rmrf(d3);
    }
    std::printf("kvstore_test OK (%d checks)\n", checks_run);
}
