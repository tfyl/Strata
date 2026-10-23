// tfyl: where the prompt is read to and checkpointed besides the turn header - the newest message however long (a
// side request forks off the history there), whether that checkpoint rotates (a tool result) or stays, and the
// branch point where a prompt leaves a path the cache has held.  Header-only, CPU.
#include "strata/program/conv_cache.hpp"
#include "strata/program/message_boundary.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace strata::program;
static int checks_run = 0;
static void check(bool ok, const char* label) {
    ++checks_run;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
int main() {
    constexpr int64_t turn = 248045, user = 846, nl = 198, tool = 248066, none = INT64_MAX;
    // A Hermes fork as the 2026-10-22 session files show it: history, then a 3,200-token user message at 38,038
    // (the side request's), then the new assistant turn.  The capped search (1,024) misses it; uncapped finds it.
    {
        std::vector<int64_t> ids(41262, 7);
        ids[0] = ids[37718] = ids[38038] = ids[41261] = turn;
        ids[38039] = user; ids[38040] = nl; ids[38041] = 58;   // "[S..."
        check(message_checkpoint_boundary(ids, 0, 41261, turn) == -1, "capped: a 3,223-token message is missed");
        check(message_checkpoint_boundary(ids, 0, 41261, turn, 8192, none) == 38038, "uncapped: its start is found");
        check(message_checkpoint_boundary(ids, 38038, 41261, turn, 8192, none) == -1, "not at or before the resume point");
        check(message_checkpoint_boundary(ids, 0, 41261, turn, 40000, none) == -1, "min_prefix still applies");
        check(!message_is_tool_result(ids, 38038, tool), "a user message is kept, not the rotating kind");
        ids[38041] = tool;
        check(message_is_tool_result(ids, 38038, tool), "<tool_response> right after the role: a tool result");
        ids[38041] = 58; ids[38040] = tool;
        check(message_is_tool_result(ids, 38038, tool), "within the role and newline tokens");
        ids[38040] = nl; ids[38042] = tool;
        check(!message_is_tool_result(ids, 38038, tool), "a tag after the first text token is text");
        check(!message_is_tool_result(ids, 38038, tool), "a quoted tag further in is text, not a tool result");
        check(!message_is_tool_result(ids, -1, tool) && !message_is_tool_result(ids, 38038, -1) &&
              !message_is_tool_result(ids, 41261, tool), "bad positions and an unset token");
    }
    // Branch points.
    {
        // the cron chats: two system prompts that differ at a timestamp 17,071 tokens in, nothing resumable before it
        check(branch_checkpoint_at(17071, 0, 18785) == 16896, "rounded down to 256 tokens");
        check(branch_checkpoint_at(17071, 0, 18785, 256) <= 17060, "a sibling a few tokens earlier still mounts it");
        // a fork 38,041 tokens in, the chain resuming from a checkpoint at 22,016
        check(branch_checkpoint_at(38041, 22016, 48494) == 37888, "a fork far past the resume point");
        // an ordinary next turn: it follows the live tokens to their end and resumes there
        check(branch_checkpoint_at(48494, 48494, 50000) == -1, "the live path continued: no branch");
        // a rewritten reply: it resumes at the turn header a few tokens before where it leaves the old path
        check(branch_checkpoint_at(41262 + 3, 41261, 42000) == -1, "a rewritten tail: not worth a state");
        check(branch_checkpoint_at(30000, 28000, 40000) == -1, "under min_gain past the resume point");
        check(branch_checkpoint_at(30000, 0, 29000) == -1, "the prompt ends before it");
        check(branch_checkpoint_at(30000, 0, 29953) == -1, "not at the prompt's last token");
        check(branch_checkpoint_at(30000, 0, 29954) == 29952, "just before it");
        check(branch_checkpoint_at(30000, 0, 40000, 0) == -1 && branch_checkpoint_at(0, 0, 40000) == -1 &&
              branch_checkpoint_at(30000, -1, 40000) == -1, "off, nothing shared, a bad resume point");
        check(branch_checkpoint_at(30000, 0, 40000, 1, 0) == 30000, "grain 1: the exact point");
    }
    // Retention: generate.cpp passes the branch point to the policy as a kept item (with the pins), so the rotating
    // newest-message and periodic checkpoints leave first.
    {
        const int64_t lens[] = {17000, 30000, 37888, 39000, 40000, 41000, 41261};
        const uint64_t stamps[] = {1, 2, 3, 4, 5, 6, 7};
        bool kept[] = {false, false, true, false, false, false, false};
        for (int round = 0; round < 3; ++round) {
            const size_t v = conv_cache::spread_victim(lens, stamps, 7, 6, nullptr, kept);
            check(v != 2 && v != 0 && v != 6, "the branch point, the root and the newest stay");
        }
        check(conv_cache::eviction_victim(stamps, 7, 6, nullptr, kept) == 1, "LRU: the oldest unkept leaf");
    }
    // Hermes' background review: the main line's history ends at 200,000, the review's "Review the conversation
    // above..." message starts there (the kept user-message point), then 8 review calls each add a tool result (the
    // rotating tail) and a turn header.  Over all of them, with --prompt-cache 6, the point the main line comes back
    // to stays - under either retention policy, and under least recent use only because of the mark.
    auto review = [](bool spread, bool marked) {
        std::vector<int64_t> lens = {15224, 180000, 200000};   // root, an older periodic one, the review's start
        std::vector<bool> keep = {false, false, marked};
        std::vector<uint64_t> stamps = {1, 2, 3};
        uint64_t clock = 3;
        int64_t at = 200050;
        for (int call = 0; call < 8; ++call) {
            for (int k = 0; k < 2; ++k) {   // a tool result's checkpoint and the turn header
                at += 300;
                lens.push_back(at); keep.push_back(false); stamps.push_back(++clock);
                while (lens.size() > 6) {
                    std::unique_ptr<bool[]> kf(new bool[keep.size()]);
                    bool any = false;
                    for (size_t i = 0; i < keep.size(); ++i) any = any || (kf[i] = keep[i]);
                    const size_t v = spread
                        ? conv_cache::spread_victim(lens.data(), stamps.data(), lens.size(), 6, nullptr, any ? kf.get() : nullptr)
                        : conv_cache::eviction_victim(stamps.data(), lens.size(), 6, nullptr, any ? kf.get() : nullptr);
                    lens.erase(lens.begin() + (std::ptrdiff_t) v);
                    keep.erase(keep.begin() + (std::ptrdiff_t) v);
                    stamps.erase(stamps.begin() + (std::ptrdiff_t) v);
                }
            }
        }
        return std::make_pair(std::find(lens.begin(), lens.end(), 200000) != lens.end(), lens.front() == 15224);
    };
    check(review(true, true).first && review(true, true).second, "spread: the review's start and the root survive");
    check(review(false, true).first && review(false, true).second, "LRU: they survive 8 calls with the mark");
    check(!review(false, false).first, "LRU without the mark: the review's start is lost (the gap it closes)");
    std::printf("branch_point_test OK (%d checks)\n", checks_run);
}
