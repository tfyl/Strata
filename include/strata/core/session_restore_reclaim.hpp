// RESTORE admission: free parked conversations before refusing a session file for RAM, never the live session.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_memory.hpp"

namespace strata::core {

struct SessionRestoreReclaimed {
    size_t parked = 0, bytes = 0;
};

// Called from the RESTORE's RAM preflight (SessionReadLimits::admit) with the parse's peak `need`, after the file's
// header and identity were checked. Probe after each eviction, as session_save_reclaim does. The retained K/V goes
// first (the restore drops it anyway), then the oldest parked conversations; pinned ones stay parked. The live
// session and its checkpoints are untouched: a refused or failed restore must leave it as it was, and they are only
// released after the transfer, so they cannot count towards the parse's peak. Unknown telemetry or an unknown
// estimate cannot justify eviction.
template<class Probe>
SessionRestoreReclaimed session_restore_reclaim(ConversationCache& cache, uint64_t need, uint64_t floor,
                                                Probe available) {
    SessionRestoreReclaimed out;
    auto room = available();
    auto done = [&] { return !room || need == UINT64_MAX || conversation_memory_admit(room, need, floor); };
    if (done()) return out;
    if (cache.retained_bytes()) {
        out.bytes += cache.retained_bytes();
        { auto unused = cache.take_reuse(); }
        room = available();
        if (done()) return out;
    }
    while (true) {
        const size_t before = cache.bytes();
        if (!cache.evict_oldest()) break; // pinned conversations stay parked
        ++out.parked;
        out.bytes += before - cache.bytes();
        room = available();
        if (done()) break;
    }
    return out;
}

} // namespace strata::core
