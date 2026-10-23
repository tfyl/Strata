// CPU only: which saves keep a single running state (--session-resume-only), what their RAM preflight asks for, and
// how far a prompt follows a parked conversation's history (the branch-point input).
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
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
int main() {
    // Which saves: the deepest checkpoint stands in for the live state when the tail past it is short.
    {
        const ConversationCheckpoint turn = checkpoint(18562);
        check(session_save_resume_only(&turn, 18785, 4096), "a 223-token reply past the turn header");
        check(session_save_resume_only(&turn, 18562, 4096), "the live state at the checkpoint itself");
        check(!session_save_resume_only(&turn, 18562 + 4097, 4096), "a tail past the limit keeps the live state");
        check(session_save_resume_only(&turn, 18562 + 4096, 4096), "exactly the limit");
        check(!session_save_resume_only(&turn, 18785, -1), "-1: never (upstream's file)");
        check(!session_save_resume_only(nullptr, 18785, 4096), "no checkpoint: the live state");
        check(!session_save_resume_only(&turn, 18000, 4096), "a checkpoint past the live tokens is not this chain's");
        ConversationCheckpoint split = checkpoint(100);
        split.stage_parts.push_back(checkpoint(100));
        check(!session_save_resume_only(&split, 120, 4096), "a layer split's checkpoint");
        ConversationCheckpoint empty;
        check(!session_save_resume_only(&empty, 10, 4096), "an empty checkpoint");
    }
    // Its preflight: one copy of the checkpoint, no live running state - less than upstream's two copies + live.
    {
        const ConversationCheckpoint deepest = checkpoint(1000);
        SessionSaveLive live;
        live.state_bytes = 118063152; live.tokens = 1200; live.images = 1; live.kv_layers = 13;
        const uint64_t one = session_save_resume_peak_bytes(deepest, live);
        const uint64_t both = session_save_peak_bytes(&deepest, live);
        check(one >= deepest.bytes() + (16u << 20), "covers the image's copy and the write buffer");
        check(both - one >= live.state_bytes + deepest.bytes(), "saves the live state and the second copy");
        SessionSaveLive huge = live;
        huge.kv_layers = UINT64_MAX;
        check(session_save_resume_peak_bytes(deepest, huge) == UINT64_MAX, "saturates");
    }
    // How far a prompt follows a parked history: the longest common prefix, same steering mode only.
    {
        ConversationCache cache(1u << 30, 4);
        SavedConversation a; a.live = checkpoint(300); a.live.ids[0] = 1; a.cvec = true;
        for (size_t i = 0; i < a.live.ids.size(); ++i) a.live.ids[i] = (int32_t) i;
        SavedConversation b = a; b.live.ids.resize(500);
        for (size_t i = 0; i < b.live.ids.size(); ++i) b.live.ids[i] = (int32_t) (i < 200 ? i : 100000 + i);
        b.cvec = false;
        check(cache.put(std::move(a)) && cache.put(std::move(b)), "fixtures parked");
        std::vector<int64_t> prompt(400);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (int64_t) (i < 250 ? i : 7);
        check(cache.longest_shared(prompt, true) == 250, "parts from the parked history at 250");
        check(cache.longest_shared(prompt, false) == 200, "the other steering mode: its own entry only");
        std::vector<int64_t> longer(1000);
        for (size_t i = 0; i < longer.size(); ++i) longer[i] = (int64_t) i;
        check(cache.longest_shared(longer, true) == 300, "follows it to its end");
        check(ConversationCache(0, 0).longest_shared(longer, true) == 0, "nothing parked");
    }
    // The branch mark travels with a layer split's parts like the pin.
    {
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(checkpoint(10));
        chain.back().branch = true;
        chain.back().stage_parts.push_back(checkpoint(10));
        auto split = conversation_checkpoints_split(std::move(chain), 1);
        check(split.stage0.size() == 1 && split.stage0[0].branch && split.parts[0][0].branch, "split keeps the mark");
    }
    std::printf("session_resume_only_test OK (%d checks)\n", checks_run);
}
