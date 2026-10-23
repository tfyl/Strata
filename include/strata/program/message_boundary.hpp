#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

inline int64_t message_checkpoint_boundary(const std::vector<int64_t>& ids, int64_t resume,
        int64_t turn_at, int64_t turn_token, int64_t min_prefix = 8192, int64_t max_tail = 1024) {
    if (resume < 0 || turn_token < 0 || turn_at <= resume || turn_at >= int64_t(ids.size()) ||
        ids[std::size_t(turn_at)] != turn_token || min_prefix < 0 || max_tail <= 0) return -1;
    for (int64_t i = turn_at - 1; i > resume; --i) {
        if (turn_at - i > max_tail) return -1;
        if (ids[std::size_t(i)] == turn_token) return i >= min_prefix ? i : -1;
    }
    return -1;
}

/// tfyl: whether the message the turn token at `at` opens is a tool result - a user turn whose text starts with
/// <tool_response> (`tool_token`), which comes within the role and newline tokens after the turn token.  An agent's
/// tool loop makes one of these per request; the user messages between them are where its side requests fork off
/// the history, and where the main line comes back to.
inline bool message_is_tool_result(const std::vector<int64_t>& ids, int64_t at, int64_t tool_token) {
    if (at < 0 || tool_token < 0) return false;
    for (int64_t i = at + 1; i <= at + 3 && i < int64_t(ids.size()); ++i)
        if (ids[std::size_t(i)] == tool_token) return true;
    return false;
}

/// tfyl: a branch point (Marconi's speculative insertion).  The prompt shares `shared` tokens with a path the cache
/// has held (the live session's tokens or a parked conversation's) but the deepest state it resumes from is at
/// `resume`: the two paths part at `shared`, a node a third request is likely to come back to (a sibling that forks
/// at the same message, a new chat whose system prompt differs only at a timestamp).  The prompt is read to there
/// and checkpointed.  Rounded down to `grain` tokens so a sibling that parts a few tokens earlier (another minute in
/// the same timestamp) still mounts it; -1 when that is not `min_gain` tokens past `resume` (not worth a 118 MB
/// state) or not before the prompt's last token.
inline int64_t branch_checkpoint_at(int64_t shared, int64_t resume, int64_t n, int64_t grain = 256,
                                    int64_t min_gain = 2048) {
    if (shared <= 0 || resume < 0 || grain <= 0 || min_gain < 0 || n < 2) return -1;
    const int64_t b = shared / grain * grain;
    if (b - resume < min_gain || b >= n - 1) return -1;
    return b;
}

} // namespace strata::program
