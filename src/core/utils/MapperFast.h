//
// Created by z00576261 on 2024/4/15.
//

#ifndef FAST_MAPPERFAST_H
#define FAST_MAPPERFAST_H

#include <atomic>
#include <mutex>
#include <algorithm>
#include <functional>
#include <string.h>

namespace RecMapper {
    constexpr int BUCKCAPACITY = 3;
    enum BuckStatus{
        BUCK_EXIST,
        BUCK_NOEXIST,
        BUCK_ERROR
    };

    enum MapperStatus{
        MAPPER_ERROR,
        MAPPER_INVALID,
        MAPPER_OK
    };

    class SpinLock {
    public:
        SpinLock() = default;
        SpinLock(const SpinLock&) = delete;
        SpinLock& operator=(const SpinLock) = delete;

        void lock() {
            while(f.test_and_set(std::memory_order_acquire));
        }

        void unlock() {
            f.clear(std::memory_order_release);
        }

    private:
        std::atomic_flag f;
    };

    struct InnerBuck{
        std::atomic<uint64_t> keys_[BUCKCAPACITY]{};
        int64_t  values_[BUCKCAPACITY]{};
        InnerBuck* next_ = nullptr;
        SpinLock spin;

        BuckStatus Insert(uint64_t, uint64_t&, std::function<bool()>);
        BuckStatus Find(uint64_t, uint64_t&);
        BuckStatus Remove(uint64_t);

    };

    class MapperFast {
    public:
        MapperFast(uint64_t cap, uint64_t res) : capacity_(cap), reserve_(res) {};

        ~MapperFast() = default;

        bool InitializeBuck();
        void UnInitializeBuck();

        MapperStatus Put(uint64_t key, uint64_t& value);

        MapperStatus Find(uint64_t key, uint64_t& value);

        MapperStatus Remove(uint64_t key);

        MapperStatus ToVector(std::vector<std::pair<uint64_t, uint64_t>>& vec);

        uint64_t Size() {
            return size_.load();
        }

    private:
        void FreeBuckMaps();
        void FreeBuckExpend();

        std::atomic<uint64_t> size_{ 0 };
        std::atomic<uint64_t> offset_{ 0 };
        uint64_t capacity_;
        uint64_t reserve_;
        uint32_t buck_count_;

        static constexpr uint32_t sub_map_count = 5;
        static constexpr uint32_t prime_max = 32;

        InnerBuck* buck_maps_[sub_map_count] {};
        InnerBuck* spec_buck = nullptr;
    };
}

#endif //FAST_MAPPERFAST_H
