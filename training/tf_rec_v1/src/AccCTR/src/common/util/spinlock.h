/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 ==============================================================================*/

#ifndef SRC_UTILS_SPINLOCK_H
#define SRC_UTILS_SPINLOCK_H

#include <atomic>
#include <mutex>
#include <thread> // NOLINT

namespace ock {
namespace ctr {
#ifdef LOCK_NOTHING

class SpinLock final {
public:
    void lock() noexcept {}
    bool try_lock() noexcept
    {
        return true;
    }
    void unlock() noexcept {}
};

#elif defined(USE_MUTEX)

class SpinLock final {
public:
    void lock() noexcept
    {
        mt_.lock();
    }
    bool try_lock() noexcept
    {
        return mt_.try_lock();
    }
    void unlock() noexcept
    {
        mt_.unlock();
    }

private:
    std::mutex mt_;
};

#else

class SpinLockG final {
public:
    const int maxSpinCountBeforeThreadYield = 64;
    SpinLockG() = default;
    SpinLockG(SpinLockG const &) = delete;
    SpinLockG(SpinLockG &&) noexcept = delete;
    SpinLockG &operator = (SpinLockG const &) = delete;
    SpinLockG &operator = (SpinLockG const &&) = delete;

    static __inline void CpuPause()
    {
#ifdef __GNUC__
#ifdef __aarch64__
        __asm volatile("yield" ::: "memory");
#elif defined(__i386__) || defined(__x86_64__)
        __asm__ __volatile__("rep;nop;nop" ::: "memory");
#else
#error "unknown architecture"
#endif
#else
#error "unknown architecture"
#endif
    }
    inline void lock() noexcept
    {
        while (lock_.exchange(true, std::memory_order_acquire)) {
            uint16_t counter = 0;
            while (lock_.load(std::memory_order_relaxed)) {
                CpuPause();
                if (++counter > maxSpinCountBeforeThreadYield) {
                    std::this_thread::yield();
                    counter = 0;
                }
            }
        }
    }

    inline bool try_lock() noexcept
    {
        if (lock_.load(std::memory_order_relaxed)) {
            return false;
        }
        return !lock_.exchange(true, std::memory_order_acquire);
    }

    inline void unlock() noexcept
    {
        lock_.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> lock_ { false };
};
}
}

#endif
#endif
