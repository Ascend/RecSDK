/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: spinlock module
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */
#ifndef SRC_UTILS_SPINLOCK_H
#define SRC_UTILS_SPINLOCK_H

#include <atomic>
#include <mutex>
#include <thread>  // NOLINT

#define DISALLOW_COPY_MOVE_AND_ASSIGN_(type) \
    type(type const &) = delete;             \
    type(type &&) noexcept = delete;         \
    type &operator=(type const &) = delete

static __inline void cpu_pause() {
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

static constexpr uint16_t g_kMaxSpinCountBeforeThreadYield = 64;

#ifdef LOCK_NOTHING

class SpinLock final {
   public:
    void lock() noexcept {}
    bool try_lock() noexcept { return true; }
    void unlock() noexcept {}
};

#elif defined(USE_MUTEX)

class SpinLock final {
public:
    void lock() noexcept { mt_.lock(); }
    bool try_lock() noexcept { return mt_.try_lock(); }
    void unlock() noexcept { mt_.unlock(); }

private:
    std::mutex mt_;
};

#else

class SpinLock final {
public:
    SpinLock() = default;

    DISALLOW_COPY_MOVE_AND_ASSIGN_(SpinLock);

    inline void lock() noexcept {
        while (true) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                break;
            }

            uint16_t counter = 0;
            while (lock_.load(std::memory_order_relaxed)) {
                cpu_pause();
                if (++counter > g_kMaxSpinCountBeforeThreadYield) {
                    std::this_thread::yield();
                    // reset counter
                    counter = 0;
                }
            }
        }
    }

    inline bool try_lock() noexcept {
        if (lock_.load(std::memory_order_relaxed)) {
            return false;
        }
        return !lock_.exchange(true, std::memory_order_acquire);
    }

    inline void unlock() noexcept { lock_.store(false, std::memory_order_release); }

private:
    std::atomic<bool> lock_{false};
};

class RWSpinLock final {
    union LockData {
        uint64_t raw;
        struct {
            uint32_t readers;
            uint32_t writer;
        } lock;
    };

public:
    RWSpinLock() = default;

    DISALLOW_COPY_MOVE_AND_ASSIGN_(RWSpinLock);

    inline void r_lock() noexcept {
        LockData oldData, newData;
        while (true) {
            uint16_t counter = 0;
            for (;;) {
                oldData.raw = lock_.load(std::memory_order_relaxed);
                if (oldData.lock.writer > 0) {
                    cpu_pause();
                    if (++counter > g_kMaxSpinCountBeforeThreadYield) {
                        std::this_thread::yield();
                        // reset counter
                        counter = 0;
                    }
                } else {
                    break;
                }
            }

            newData.lock.readers = oldData.lock.readers + 1;
            newData.lock.writer = 0;
            if (lock_.compare_exchange_weak(oldData.raw, newData.raw,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                break;
            }
        }
    }

    inline void w_lock() noexcept {
        LockData oldData, newData;
        while (true) {
            uint16_t counter = 0;
            for (;;) {
                oldData.raw = lock_.load(std::memory_order_relaxed);
                if (oldData.raw != 0) {
                    cpu_pause();
                    if (++counter > g_kMaxSpinCountBeforeThreadYield) {
                        std::this_thread::yield();
                        // reset counter
                        counter = 0;
                    }
                } else {
                    break;
                }
            }

            newData.lock.readers = 0;
            newData.lock.writer = 1;
            if (lock_.compare_exchange_weak(oldData.raw, newData.raw,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                break;
            }
        }
    }

    inline void r_unlock() noexcept { --lock_; }

    inline void w_unlock() noexcept { lock_.store(0, std::memory_order_release); }

private:
    std::atomic<uint64_t> lock_{0};
};

#endif
#endif
