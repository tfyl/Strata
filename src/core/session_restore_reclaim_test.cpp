// CPU only: pressure is injected through telemetry, not by exhausting the host.
#include "strata/core/session_restore_reclaim.hpp"
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
static int checks_run = 0;
static void check(bool ok, const char* label) {
    ++checks_run;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
static ConversationCheckpoint checkpoint(size_t n, bool pinned = false) {
    ConversationCheckpoint c;
    c.ids.assign(n, 42); c.imgs = {{1, 99}};
    c.gdn.assign(4096, 1); c.ple = {2}; c.tails = {3}; c.dead = {4}; c.block_pos = {5};
    c.used = n; c.pinned = pinned;
    return c;
}
static void park(ConversationCache& cache, int tokens, bool pin = false) {
    SavedConversation c; c.live = checkpoint(tokens);
    c.checkpoints.push_back(checkpoint(tokens - 1, pin));
    // Distinct histories, so put() does not remove superseded entries.
    c.live.ids[0] = tokens; c.checkpoints[0].ids[0] = tokens;
    check(cache.put(std::move(c)), "fixture parked");
}
int main() {
    const uint64_t need = 2000ull << 20, floor = 2048ull << 20;
    // Persistent pressure: retained K/V, then every unpinned parked conversation, oldest first; pins stay.
    {
        ConversationCache cache(1u << 20, 10);
        park(cache, 10); park(cache, 20, true); park(cache, 30);
        std::vector<ConversationKv> kv(1); kv[0].k.resize(16384); cache.retain(std::move(kv), 100);
        const size_t bytes_before = cache.bytes();
        auto freed = session_restore_reclaim(cache, need, floor, [] { return std::optional<uint64_t>(0); });
        check(freed.parked == 2 && cache.size() == 1 && cache.longest_tokens() == 20, "pinned conversation stays parked");
        check(cache.evictions() == 2 && cache.retained_bytes() == 0, "cache accounting and retained buffers");
        check(freed.bytes == bytes_before - cache.bytes(), "released byte accounting");
        auto again = session_restore_reclaim(cache, need, floor, [] { return std::optional<uint64_t>(0); });
        check(!again.bytes && !again.parked && cache.size() == 1, "exhausted reclaim is idempotent");
    }
    // No pressure, unknown telemetry, unknown estimate: nothing evicted.
    for (int mode = 0; mode < 3; ++mode) {
        ConversationCache cache(1u << 20, 10); park(cache, 10); park(cache, 20);
        std::vector<ConversationKv> kv(1); kv[0].v.resize(100); cache.retain(std::move(kv), 100);
        const auto bytes = cache.bytes();
        auto r = session_restore_reclaim(cache, mode == 2 ? UINT64_MAX : need, floor, [&]() -> std::optional<uint64_t> {
            if (mode == 1) return std::nullopt;
            return mode == 0 ? need + floor : 0;
        });
        check(!r.bytes && !r.parked && cache.bytes() == bytes && cache.size() == 2, "no eviction without known pressure");
    }
    // The measured example: 1985 MiB + 2048 MiB floor with 2716 MiB available, one 1781 MiB conversation parked.
    // Evicting the oldest suffices: the newer one stays parked.
    for (int sufficient_after = 1; sufficient_after <= 3; ++sufficient_after) {
        ConversationCache cache(1u << 20, 10); park(cache, 10); park(cache, 20);
        std::vector<ConversationKv> kv(1); kv[0].v.resize(100); cache.retain(std::move(kv), 100);
        const uint64_t want = 1985ull << 20;
        int probes = 0;
        auto r = session_restore_reclaim(cache, want, floor, [&]() -> std::optional<uint64_t> {
            return probes++ >= sufficient_after ? (2716ull + 1781ull) << 20 : 2716ull << 20;
        });
        check(probes == sufficient_after + 1, "stop immediately once RAM meets the unchanged floor");
        check(r.parked == size_t(sufficient_after - 1), "one parked conversation at a time");
        check(cache.size() == size_t(3 - sufficient_after), "only what was needed is evicted");
        if (sufficient_after == 2) check(cache.longest_tokens() == 20, "oldest goes first");
        check(conversation_memory_admit((2716ull + 1781ull) << 20, want, floor), "the example fits after eviction");
    }
    {
        ConversationCache cache(0, 0);
        auto r = session_restore_reclaim(cache, need, floor, [] { return std::optional<uint64_t>(0); });
        check(!r.bytes && !r.parked, "disabled cache");
    }
    std::printf("session_restore_reclaim: %d checks passed\n", checks_run);
}
