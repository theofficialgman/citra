// Copyright 2013 Dolphin Emulator Project / 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

#include "common/flag.h"

namespace Common {

class Event final {
public:
    void Set() {
        if (flag.TestAndSet()) {
            // Lock and immediately unlock m_mutex.
            {
                // Holding the lock at any time between the change of our flag and notify call
                // is sufficient to prevent a race where both of these actions
                // happen between the other thread's predicate test and wait call
                // which would cause wait to block until the next spurious wakeup or timeout.

                // Unlocking before notification is a micro-optimization to prevent
                // the notified thread from immediately blocking on the mutex.
                std::lock_guard<std::mutex> lk(mutex);
            }

            condvar.notify_one();
        }
    }

    void Wait() {
        if (flag.TestAndClear()) {
            return;
        }

        std::unique_lock<std::mutex> lk(mutex);
        condvar.wait(lk, [&] { return flag.TestAndClear(); });
    }

    template <class Rep, class Period>
    bool WaitFor(const std::chrono::duration<Rep, Period>& rel_time) {
        if (flag.TestAndClear())
            return true;

        std::unique_lock<std::mutex> lk(mutex);
        bool signaled = condvar.wait_for(lk, rel_time, [&] { return flag.TestAndClear(); });

        return signaled;
    }

    void Reset() {
        // no other action required, since wait loops on
        // the predicate and any lingering signal will get
        // cleared on the first iteration
        flag.Clear();
    }

private:
    Flag flag;
    std::condition_variable condvar;
    std::mutex mutex;
};

class Barrier {
public:
    explicit Barrier(std::size_t count_) : count(count_) {}

    /// Blocks until all "count" threads have called Sync()
    void Sync() {
        std::unique_lock lk{mutex};
        const std::size_t current_generation = generation;

        if (++waiting == count) {
            generation++;
            waiting = 0;
            condvar.notify_all();
        } else {
            condvar.wait(lk,
                         [this, current_generation] { return current_generation != generation; });
        }
    }

    std::size_t Generation() const {
        std::unique_lock lk(mutex);
        return generation;
    }

private:
    std::condition_variable condvar;
    mutable std::mutex mutex;
    std::size_t count;
    std::size_t waiting = 0;
    std::size_t generation = 0; // Incremented once each time the barrier is used
};

void SetCurrentThreadName(const char* name);

} // namespace Common
