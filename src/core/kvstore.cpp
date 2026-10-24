// tfyl: the conversation cache's disk tier (kvstore.hpp).
#include "strata/core/kvstore.hpp"
#include "strata/core/conversation_file.hpp"   // SessionHasher

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#if defined(_WIN32)
// The disk tier is POSIX (pread/pwrite, O_DIRECT, fdatasync): on Windows the store never opens and --kv-store is off.
namespace strata::core::kvstore {
Key chain_seed(const std::vector<uint64_t>& identity) {
    SessionHasher a(0x6b7673656564612eull), b(0x6b76736565646221ull);
    a.update(identity.data(), identity.size() * 8); b.update(identity.data(), identity.size() * 8);
    return {a.digest(), b.digest()};
}
Key chain_next(const Key& prev, const int32_t* tokens, size_t n) {
    SessionHasher a(prev.a), b(prev.b);
    a.update(tokens, n * 4); b.update(tokens, n * 4);
    return {a.digest(), b.digest()};
}
std::vector<Key> chain_keys(const Key& seed, const int32_t*, size_t) { return {seed}; }
uint64_t digest(const std::vector<Piece>&) { return 0; }
uint64_t chain_lineage(uint64_t prev, uint64_t sum) { return prev ^ sum; }
struct Store::Index {};
Store::Store() {}
Store::~Store() {}
bool Store::open(const Options&, std::string& error) { error = "kv store: not on Windows"; return false; }
const Record* Store::chunk(const Key&) const { return nullptr; }
std::vector<const Record*> Store::snapshots_at(const Key&) const { return {}; }
const Record* Store::best_snapshot(const std::vector<Key>&, const int32_t*, size_t) const { return nullptr; }
std::vector<const Record*> Store::snapshots_on_path(const std::vector<Key>&, const int32_t*, size_t) const { return {}; }
size_t Store::stored_chunks(const std::vector<Key>&) const { return 0; }
bool Store::put_chunk(const Key&, const Key&, int32_t, const std::vector<Piece>&, std::string& e, bool*, bool) {
    e = "off"; return false;
}
bool Store::put_snapshot(const Key&, int64_t, const std::vector<int32_t>&, uint32_t, const std::vector<Piece>&,
                         std::string& e, uint64_t) {
    e = "off"; return false;
}
bool Store::read(const Record&, void*, size_t, std::string& e) { e = "off"; return false; }
bool Store::read(const Record&, const std::vector<Out>&, std::string& e) { e = "off"; return false; }
void Store::touch(const Record&) {}
bool Store::sync(std::string&) { return true; }
}  // namespace strata::core::kvstore
#else

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::core::kvstore {
namespace {
constexpr uint64_t kMagic = 0x31305453564b5653ull;   // "SVKVST01"
constexpr size_t kBounce = 16u << 20;
constexpr size_t kHeaderSum = kPage - 8;             // the header digest covers the bytes before it

void put64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }
void put32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
uint64_t get64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
uint32_t get32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint64_t padded(uint64_t n) { return (n + kPage - 1) / kPage * kPage; }

bool fail(std::string& error, const std::string& what, int e = 0) {
    error = "kv store: " + what + (e ? std::string(": ") + std::strerror(e) : std::string());
    return false;
}

void encode(const Record& r, uint8_t* h) {
    std::memset(h, 0, kPage);
    put64(h, kMagic);
    put32(h + 8, (uint32_t) r.kind);
    put32(h + 12, r.flags);
    put64(h + 16, r.key.a); put64(h + 24, r.key.b);
    put64(h + 32, r.parent.a); put64(h + 40, r.parent.b);
    put64(h + 48, r.payload); put64(h + 56, r.sum); put64(h + 64, r.seq);
    put32(h + 72, (uint32_t) r.next_token);
    put32(h + 76, (uint32_t) r.tail.size());
    put64(h + 80, (uint64_t) r.tokens);
    put64(h + 1112, r.lineage);   // past the longest tail (88 + 255 * 4); 0 in records written before tfyl.6
    if (!r.tail.empty()) std::memcpy(h + 88, r.tail.data(), r.tail.size() * 4);
    put64(h + kHeaderSum, session_hash64(h, kHeaderSum, kMagic));
}

bool decode(const uint8_t* h, Record& r) {
    if (get64(h) != kMagic || get64(h + kHeaderSum) != session_hash64(h, kHeaderSum, kMagic)) return false;
    const uint32_t kind = get32(h + 8), tail_n = get32(h + 76);
    if ((kind != (uint32_t) Kind::chunk && kind != (uint32_t) Kind::snapshot) || tail_n > kMaxTail) return false;
    r.kind = (Kind) kind;
    r.flags = get32(h + 12);
    r.key = {get64(h + 16), get64(h + 24)};
    r.parent = {get64(h + 32), get64(h + 40)};
    r.payload = get64(h + 48); r.sum = get64(h + 56); r.seq = get64(h + 64);
    r.next_token = (int32_t) get32(h + 72);
    r.tokens = (int64_t) get64(h + 80);
    r.lineage = get64(h + 1112);
    r.tail.assign(tail_n, 0);
    if (tail_n) std::memcpy(r.tail.data(), h + 88, tail_n * 4);
    return true;
}

bool pwrite_all(int fd, const void* p, size_t n, uint64_t off) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    while (n) {
        const ssize_t w = ::pwrite(fd, b, n, (off_t) off);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        b += w; n -= (size_t) w; off += (uint64_t) w;
    }
    return true;
}
bool pread_all(int fd, void* p, size_t n, uint64_t off) {
    uint8_t* b = static_cast<uint8_t*>(p);
    while (n) {
        const ssize_t r = ::pread(fd, b, n, (off_t) off);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) { errno = EIO; return false; }
        b += r; n -= (size_t) r; off += (uint64_t) r;
    }
    return true;
}
}  // namespace

Key chain_seed(const std::vector<uint64_t>& identity) {
    SessionHasher a(0x6b7673656564612eull), b(0x6b76736565646221ull);
    a.update(identity.data(), identity.size() * 8);
    b.update(identity.data(), identity.size() * 8);
    return {a.digest(), b.digest()};
}

Key chain_next(const Key& prev, const int32_t* tokens, size_t n) {
    SessionHasher a(prev.a ^ 0x636861696e6e2d61ull), b(prev.b ^ 0x636861696e6e2d62ull);
    const uint64_t hdr[3] = {prev.a, prev.b, (uint64_t) n};
    a.update(hdr, sizeof hdr); b.update(hdr, sizeof hdr);
    a.update(tokens, n * 4); b.update(tokens, n * 4);
    return {a.digest(), b.digest()};
}

std::vector<Key> chain_keys(const Key& seed, const int32_t* tokens, size_t n) {
    std::vector<Key> keys;
    keys.reserve(n / (size_t) kChunkTokens + 1);
    keys.push_back(seed);
    for (size_t j = 0; (j + 1) * (size_t) kChunkTokens <= n; ++j)
        keys.push_back(chain_next(keys.back(), tokens + j * (size_t) kChunkTokens, (size_t) kChunkTokens));
    return keys;
}

uint64_t chain_lineage(uint64_t prev, uint64_t chunk_sum) {
    const uint64_t v[2] = {prev, chunk_sum};
    const uint64_t h = session_hash64(v, sizeof v, 0x6c696e6561676532ull);
    return h ? h : 1;   // 0 is "unknown"
}

uint64_t digest(const std::vector<Piece>& payload) {
    SessionHasher h(0x4b5653544f524531ull);
    for (const Piece& p : payload) h.update(p.data, p.n);
    return h.digest();
}

struct Store::Index {
    std::unordered_map<Key, Record, KeyHash> chunks;
    std::unordered_map<Key, std::vector<Record>, KeyHash> snaps;   // by the key of the chunks below them
    std::map<uint32_t, uint64_t> segs;                             // segment -> bytes
};

Store::Store() : ix_(new Index) {}
Store::~Store() {
    if (fd_ >= 0) {
        if (end_) ::fdatasync(fd_);
        ::close(fd_);
    }
    std::free(bounce_);
}

std::string Store::seg_path(uint32_t seg) const {
    char name[32];
    std::snprintf(name, sizeof name, "/seg-%08u.kvs", seg);
    return opt_.dir + name;
}

void Store::index_put(Record&& r) {
    if (r.kind == Kind::chunk) {
        auto it = ix_->chunks.find(r.key);
        if (it != ix_->chunks.end()) {
            if (it->second.seq > r.seq) return;
            --st_.chunks; --st_.records;
        }
        ++st_.chunks; ++st_.records;
        ix_->chunks[r.key] = std::move(r);
        return;
    }
    auto& v = ix_->snaps[r.key];
    for (auto& s : v)
        if (s.tail == r.tail) {
            if (s.seq > r.seq) return;
            s = std::move(r);
            return;
        }
    ++st_.snapshots; ++st_.records;
    v.push_back(std::move(r));
}

void Store::index_drop(const Record& r) {
    if (r.kind == Kind::chunk) {
        auto it = ix_->chunks.find(r.key);
        if (it != ix_->chunks.end() && it->second.seq == r.seq) { ix_->chunks.erase(it); --st_.chunks; --st_.records; }
        return;
    }
    auto it = ix_->snaps.find(r.key);
    if (it == ix_->snaps.end()) return;
    auto& v = it->second;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].seq == r.seq) { v.erase(v.begin() + (std::ptrdiff_t) i); --st_.snapshots; --st_.records; break; }
    if (v.empty()) ix_->snaps.erase(it);
}

bool Store::scan(uint32_t seg, std::string& error) {
    const int fd = ::open(seg_path(seg).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail(error, "cannot open " + seg_path(seg), errno);
    struct stat sb{};
    ::fstat(fd, &sb);
    const uint64_t size = (uint64_t) sb.st_size;
    std::vector<uint8_t> h(kPage);
    uint64_t off = 0;
    while (off + kPage <= size) {
        Record r;
        if (!pread_all(fd, h.data(), kPage, off) || !decode(h.data(), r) || off + r.bytes_on_disk() > size) break;
        r.seg = seg; r.off = off;
        seq_ = std::max(seq_, r.seq + 1);
        off += r.bytes_on_disk();
        index_put(std::move(r));
    }
    ::close(fd);
    if (off < size) ++st_.torn;   // a torn or foreign tail: what follows the last good record is not indexed
    ix_->segs[seg] = off;
    if (seg == seg_) end_ = off;
    return true;
}

bool Store::open(const Options& o, std::string& error) {
    opt_ = o;
    if (opt_.dir.empty()) return fail(error, "no directory");
    if (opt_.segment_bytes < 4 * kPage) opt_.segment_bytes = 4 * kPage;
    if (::mkdir(opt_.dir.c_str(), 0700) != 0 && errno != EEXIST) return fail(error, "cannot create " + opt_.dir, errno);
    if (!bounce_) {
        void* p = nullptr;
        if (posix_memalign(&p, kPage, kBounce) != 0) return fail(error, "no memory for the I/O buffer");
        bounce_ = static_cast<uint8_t*>(p);
        bounce_n_ = kBounce;
    }
    std::vector<uint32_t> segs;
    if (DIR* d = ::opendir(opt_.dir.c_str())) {
        while (const dirent* e = ::readdir(d)) {
            unsigned n = 0;
            char tail = 0;
            if (std::sscanf(e->d_name, "seg-%8u.kvs%c", &n, &tail) == 1 && std::strlen(e->d_name) == 16) segs.push_back(n);
        }
        ::closedir(d);
    } else {
        return fail(error, "cannot list " + opt_.dir, errno);
    }
    std::sort(segs.begin(), segs.end());
    seg_ = segs.empty() ? 1 : segs.back();
    for (uint32_t s : segs)
        if (!scan(s, error)) return false;
    if (segs.empty() || end_ >= opt_.segment_bytes) {
        if (!segs.empty()) ++seg_;
        end_ = 0;
    }
    const std::string path = seg_path(seg_);
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd_ < 0) return fail(error, "cannot open " + path, errno);
    if (::ftruncate(fd_, (off_t) end_) != 0) return fail(error, "cannot truncate " + path, errno);
    direct_ = false;
#ifdef O_DIRECT
    if (opt_.direct) {
        const int fl = ::fcntl(fd_, F_GETFL);
        direct_ = fl >= 0 && ::fcntl(fd_, F_SETFL, fl | O_DIRECT) == 0;
    }
#endif
    ix_->segs[seg_] = end_;
    st_.segments = ix_->segs.size();
    st_.disk_bytes = 0;
    for (const auto& s : ix_->segs) st_.disk_bytes += s.second;
    return keep_budget(error);
}

bool Store::roll(std::string& error) {
    if (::fdatasync(fd_) != 0) return fail(error, "cannot flush " + seg_path(seg_), errno);
    ::close(fd_);
    ++seg_;
    end_ = 0;
    const std::string path = seg_path(seg_);
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd_ < 0) return fail(error, "cannot create " + path, errno);
#ifdef O_DIRECT
    if (direct_) {
        const int fl = ::fcntl(fd_, F_GETFL);
        direct_ = fl >= 0 && ::fcntl(fd_, F_SETFL, fl | O_DIRECT) == 0;
    }
#endif
    ix_->segs[seg_] = 0;
    st_.segments = ix_->segs.size();
    return true;
}

bool Store::append(Record& r, const std::vector<Piece>& payload, std::string& error) {
    if (fd_ < 0) return fail(error, "not open");
    r.payload = 0;
    for (const Piece& p : payload) r.payload += p.n;
    r.sum = digest(payload);
    r.seq = seq_++;
    r.used = 0;
    if (end_ > 0 && end_ + r.bytes_on_disk() > opt_.segment_bytes && !roll(error)) return false;
    r.seg = seg_;
    r.off = end_;
    // the payload first, the header last: a crash in between leaves no valid header, so the scan drops it
    uint64_t at = end_ + kPage;
    size_t fill = 0;
    auto flush = [&](bool last) -> bool {
        size_t n = fill;
        if (last) { const size_t p = (size_t) padded(n); std::memset(bounce_ + n, 0, p - n); n = p; }
        if (!n) return true;
        if (!pwrite_all(fd_, bounce_, n, at)) return fail(error, "cannot write " + seg_path(seg_), errno);
        at += n;
        fill = 0;
        return true;
    };
    for (const Piece& p : payload) {
        const uint8_t* src = static_cast<const uint8_t*>(p.data);
        size_t left = p.n;
        while (left) {
            const size_t c = std::min(left, bounce_n_ - fill);
            std::memcpy(bounce_ + fill, src, c);
            fill += c; src += c; left -= c;
            if (fill == bounce_n_ && !flush(false)) return false;
        }
    }
    if (!flush(true)) return false;
    encode(r, bounce_);
    if (!pwrite_all(fd_, bounce_, kPage, end_)) return fail(error, "cannot write " + seg_path(seg_), errno);
    end_ += r.bytes_on_disk();
    ix_->segs[seg_] = end_;
    st_.disk_bytes += r.bytes_on_disk();
    st_.written_bytes += r.bytes_on_disk();
    return true;
}

bool Store::keep_budget(std::string& error) {
    while (st_.disk_bytes > opt_.max_bytes && ix_->segs.size() > 1) {
        const uint32_t old = ix_->segs.begin()->first;
        if (old == seg_) break;
        // the records still indexed in it: the ones used since they were written get one more segment's life
        std::vector<Record> live;
        for (const auto& c : ix_->chunks) if (c.second.seg == old) live.push_back(c.second);
        for (const auto& v : ix_->snaps) for (const auto& s : v.second) if (s.seg == old) live.push_back(s);
        const int fd = ::open(seg_path(old).c_str(), O_RDONLY | O_CLOEXEC);
        for (const Record& r : live) {
            index_drop(r);
            if (r.used == 0 || fd < 0) { ++st_.gc_dropped; continue; }
            // copy forward in pieces of the I/O buffer: the payload, then its header (same digest, a new place)
            Record c = r;
            c.used = 0;
            c.seq = seq_++;
            if (end_ > 0 && end_ + c.bytes_on_disk() > opt_.segment_bytes && !roll(error)) { ::close(fd); return false; }
            c.seg = seg_; c.off = end_;
            uint64_t left = padded(r.payload), from = r.off + kPage, to = end_ + kPage;
            bool ok = true;
            while (left && ok) {
                const size_t n = (size_t) std::min<uint64_t>(left, bounce_n_);
                ok = pread_all(fd, bounce_, n, from) && pwrite_all(fd_, bounce_, n, to);
                left -= n; from += n; to += n;
            }
            if (ok) { encode(c, bounce_); ok = pwrite_all(fd_, bounce_, kPage, end_); }
            if (!ok) { ++st_.gc_dropped; continue; }
            end_ += c.bytes_on_disk();
            ix_->segs[seg_] = end_;
            st_.disk_bytes += c.bytes_on_disk();
            st_.written_bytes += c.bytes_on_disk();
            ++st_.gc_copied;
            index_put(std::move(c));
        }
        if (fd >= 0) ::close(fd);
        st_.disk_bytes -= std::min(st_.disk_bytes, ix_->segs.begin()->second);
        ix_->segs.erase(ix_->segs.begin());
        ::unlink(seg_path(old).c_str());
        ++st_.gc_segments;
        st_.segments = ix_->segs.size();
    }
    return true;
}

const Record* Store::chunk(const Key& k) const {
    auto it = ix_->chunks.find(k);
    return it == ix_->chunks.end() ? nullptr : &it->second;
}

std::vector<const Record*> Store::snapshots_at(const Key& at) const {
    std::vector<const Record*> out;
    auto it = ix_->snaps.find(at);
    if (it != ix_->snaps.end())
        for (const auto& s : it->second) out.push_back(&s);
    return out;
}

size_t Store::stored_chunks(const std::vector<Key>& keys) const {
    size_t j = 0;
    while (j + 1 < keys.size()) {
        const Record* c = chunk(keys[j + 1]);
        if (!c || c->parent != keys[j]) break;
        ++j;
    }
    return j;
}

const Record* Store::best_snapshot(const std::vector<Key>& keys, const int32_t* prompt, size_t n) const {
    if (keys.empty() || n < 2) return nullptr;
    const size_t whole = stored_chunks(keys);
    for (size_t j = whole + 1; j-- > 0;) {
        const Record* best = nullptr;
        for (const Record* s : snapshots_at(keys[j])) {
            const size_t base = j * (size_t) kChunkTokens, len = base + s->tail.size();
            if ((int64_t) len != s->tokens || len >= n) continue;
            if (!std::equal(s->tail.begin(), s->tail.end(), prompt + base)) continue;
            if (!best || s->tokens > best->tokens) best = s;
        }
        if (best) return best;
    }
    return nullptr;
}

std::vector<const Record*> Store::snapshots_on_path(const std::vector<Key>& keys, const int32_t* prompt,
                                                    size_t n) const {
    std::vector<const Record*> out;
    if (keys.empty() || n < 2) return out;
    const size_t whole = stored_chunks(keys);
    for (size_t j = whole + 1; j-- > 0;) {
        const size_t first = out.size();
        for (const Record* s : snapshots_at(keys[j])) {
            const size_t base = j * (size_t) kChunkTokens, len = base + s->tail.size();
            if ((int64_t) len != s->tokens || len >= n) continue;
            if (!std::equal(s->tail.begin(), s->tail.end(), prompt + base)) continue;
            out.push_back(s);
        }
        std::sort(out.begin() + (std::ptrdiff_t) first, out.end(),
                  [](const Record* a, const Record* b) { return a->tokens > b->tokens; });
    }
    return out;
}

bool Store::put_chunk(const Key& k, const Key& parent, int32_t next_token, const std::vector<Piece>& payload,
                      std::string& error, bool* written, bool replace) {
    if (written) *written = false;
    if (chunk(k) && !replace) return true;
    Record r;
    r.kind = Kind::chunk;
    r.key = k; r.parent = parent; r.next_token = next_token;
    if (!append(r, payload, error)) return false;
    ++st_.chunks_written;
    index_put(std::move(r));
    if (written) *written = true;
    return keep_budget(error);
}

bool Store::put_snapshot(const Key& at, int64_t tokens, const std::vector<int32_t>& tail, uint32_t flags,
                         const std::vector<Piece>& payload, std::string& error, uint64_t lineage) {
    if (tail.size() > kMaxTail) return fail(error, "snapshot tail longer than a chunk");
    Record r;
    r.kind = Kind::snapshot;
    r.key = at; r.tokens = tokens; r.tail = tail; r.flags = flags; r.lineage = lineage;
    if (!append(r, payload, error)) return false;
    ++st_.snapshots_written;
    index_put(std::move(r));
    return keep_budget(error);
}

bool Store::read(const Record& r, void* dst, size_t n, std::string& error) {
    return read(r, std::vector<Out>{{dst, n}}, error);
}

bool Store::read(const Record& r, const std::vector<Out>& dst, std::string& error) {
    size_t n = 0;
    for (const Out& d : dst) n += d.n;
    if (n != r.payload) return fail(error, "payload size mismatch");
    int fd = -1;
    if (r.seg != seg_) {
        fd = ::open(seg_path(r.seg).c_str(), O_RDONLY | O_CLOEXEC);
#ifdef O_DIRECT
        if (fd >= 0 && direct_) {   // a sealed segment, read past the page cache too
            const int fl = ::fcntl(fd, F_GETFL);
            if (fl >= 0) (void) ::fcntl(fd, F_SETFL, fl | O_DIRECT);
        }
#endif
    }
    const int use = r.seg == seg_ ? fd_ : fd;
    if (use < 0) { ++st_.read_errors; return fail(error, "cannot open " + seg_path(r.seg), errno); }
    SessionHasher h(0x4b5653544f524531ull);
    uint64_t left = padded(r.payload), from = r.off + kPage, got = 0;
    size_t di = 0, dat = 0;   // the destination piece and the offset in it
    bool ok = true;
    while (left && ok) {
        const size_t c = (size_t) std::min<uint64_t>(left, bounce_n_);
        ok = pread_all(use, bounce_, c, from);
        if (ok) {
            size_t take = (size_t) std::min<uint64_t>(c, r.payload - got), at = 0;
            h.update(bounce_, take);
            got += take;
            while (take) {
                while (di < dst.size() && dat == dst[di].n) { ++di; dat = 0; }
                const size_t k = std::min(take, dst[di].n - dat);
                std::memcpy(static_cast<uint8_t*>(dst[di].data) + dat, bounce_ + at, k);
                dat += k; at += k; take -= k;
            }
        }
        left -= c; from += c;
    }
    const int e = errno;
    if (fd >= 0) ::close(fd);
    if (!ok) { ++st_.read_errors; return fail(error, "cannot read " + seg_path(r.seg), e); }
    if (h.digest() != r.sum) { ++st_.read_errors; return fail(error, "payload digest mismatch in " + seg_path(r.seg)); }
    touch(r);
    return true;
}

void Store::touch(const Record& r) { const_cast<Record&>(r).used = ++clock_; }

bool Store::sync(std::string& error) {
    if (fd_ >= 0 && end_ && ::fdatasync(fd_) != 0) return fail(error, "cannot flush " + seg_path(seg_), errno);
    return true;
}

}  // namespace strata::core::kvstore

#endif  // !_WIN32
