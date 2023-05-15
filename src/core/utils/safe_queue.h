/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description:  safe queue class
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef SAFE_QUEUE_H
#define SAFE_QUEUE_H

#include <queue>
#include <memory>
#include <mutex>
#include <string>
#include <condition_variable>
#include <list>
#include "common.h"

template<class T>
class SafeQueue {
    static constexpr uint64_t DEFAULT_CAP = 10;

public:
    SafeQueue() = default;

    ~SafeQueue() = default;

    SafeQueue(SafeQueue const& other)
    {
        std::lock_guard<std::mutex> lk(other.mut);
        dataQueue = other.dataQueue;
    }

    SafeQueue& operator=(SafeQueue const& other)
    {
        if (this == &other) {
            return *this;
        }
        std::lock_guard<std::mutex> lk(other.mut);
        dataQueue = other.dataQueue;
        return *this;
    }

    std::unique_ptr<T> GetOne()
    {
        std::lock_guard<std::mutex> lk(mut);
        if (emptyQueue.empty()) {
            return std::make_unique<T>();
        } else {
            auto t = move(emptyQueue.back());
            emptyQueue.pop_back();
            return move(t);
        }
    }

    std::unique_ptr<T> WaitAndGetOne()
    {
        {
            std::lock_guard<std::mutex> lk(mut);
            if (creatNum < capacity) {
                creatNum++;
                return std::make_unique<T>();
            }
        }
        std::unique_lock<std::mutex> locker(mut);
        dirtyCond.wait(locker, [this] { return !emptyQueue.empty(); });
        auto t = move(emptyQueue.back());
        emptyQueue.pop_back();
        return move(t);
    }

    void PutDirty(std::unique_ptr<T>&& t)
    {
        std::lock_guard<std::mutex> lk(mut);
        emptyQueue.push_back(move(t));
        dirtyCond.notify_one();
    }

    void Pushv(std::unique_ptr<T>&& t) // 入队操作
    {
        std::lock_guard<std::mutex> lk(mut);
        dataQueue.push_back(move(t));
        dataCond.notify_one();
    }

    std::unique_ptr<T> WaitAndPop()
    {
        std::unique_lock<std::mutex> lk(mut);
        dataCond.wait(lk, [this] { return !dataQueue.empty(); });
        std::unique_ptr<T> res = std::move(dataQueue.front());
        dataQueue.pop_front();
        return move(res);
    }

    std::unique_ptr<T> TryPop()
    {
        std::lock_guard<std::mutex> lk(mut);
        if (dataQueue.empty()) {
            return nullptr;
        }
        std::unique_ptr<T> res = std::move(dataQueue.front());
        dataQueue.pop_front();
        return move(res);
    }

    bool Empty() const
    {
        std::lock_guard<std::mutex> lk(mut);
        return dataQueue.empty();
    }

    size_t Size() const
    {
        std::lock_guard<std::mutex> lk(mut);
        return dataQueue.size();
    }

private:
    mutable std::mutex mut;
    uint64_t capacity = DEFAULT_CAP;
    std::atomic<uint64_t> creatNum {};
    std::list<std::unique_ptr<T>> dataQueue;
    std::list<std::unique_ptr<T>> emptyQueue;
    std::condition_variable dataCond;
    std::condition_variable dirtyCond;
};

template<class T>
class SingletonQueue {
public:
    static SafeQueue<T>* getInstances(int i)
    {
        static SafeQueue<T> instance[MxRec::MAX_QUEUE_NUM];
        if (i >= MxRec::MAX_QUEUE_NUM || i < 0) {
            return nullptr;
        }
        return &instance[i];
    };

    SingletonQueue() = delete;

    ~SingletonQueue() = delete;

    SingletonQueue(T&&) = delete;

    SingletonQueue(const T&) = delete;

    void operator=(const T&) = delete;
};

#endif