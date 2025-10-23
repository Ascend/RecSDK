/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef SAFE_QUEUE_H
#define SAFE_QUEUE_H

#include <queue>
#include <memory>
#include <mutex>
#include <string>
#include <condition_variable>
#include <list>
#include "common.h"
namespace MxRec {
    template<class T>
    class SafeQueue {
        static constexpr uint64_t DEFAULT_CAP = 10;

    public:
        SafeQueue() = default;

        ~SafeQueue() = default;

        SafeQueue(SafeQueue const &other)
        {
            std::lock_guard <std::mutex> lk(other.mut);
            dataQueue = other.dataQueue;
        }

        SafeQueue &operator=(SafeQueue const &other)
        {
            if (this == &other) {
                return *this;
            }
            std::lock_guard <std::mutex> lk(other.mut);
            dataQueue = other.dataQueue;
            return *this;
        }

        std::unique_ptr <T> GetOne()
        {
            std::lock_guard <std::mutex> lk(mut);
            if (emptyQueue.empty()) {
                return std::make_unique<T>();
            } else {
                auto t = move(emptyQueue.back());
                emptyQueue.pop_back();
                return move(t);
            }
        }

        void PutDirty(std::unique_ptr <T> &&t)
        {
            std::lock_guard <std::mutex> lk(mut);
            emptyQueue.push_back(move(t));
            dirtyCond.notify_one();
        }

        void Pushv(std::unique_ptr <T> &&t) // 入队操作
        {
            std::lock_guard <std::mutex> lk(mut);
            dataQueue.push_back(move(t));
            dataCond.notify_one();
        }

        std::unique_ptr <T> TryPop()
        {
            std::lock_guard <std::mutex> lk(mut);
            if (dataQueue.empty()) {
                return nullptr;
            }
            std::unique_ptr <T> res = std::move(dataQueue.front());
            dataQueue.pop_front();
            return move(res);
        }

        size_t Size() const
        {
            std::lock_guard <std::mutex> lk(mut);
            return dataQueue.size();
        }

    private:
        mutable std::mutex mut;
        uint64_t capacity = DEFAULT_CAP;
        std::list <std::unique_ptr<T>> dataQueue;
        std::list <std::unique_ptr<T>> emptyQueue;
        std::condition_variable dataCond;
        std::condition_variable dirtyCond;
    };

    template<class T>
    class SingletonQueue {
    public:
        static SafeQueue<T> *GetInstances(int i)
        {
            static SafeQueue<T> instance[MxRec::MAX_QUEUE_NUM];
            if (i >= MxRec::MAX_QUEUE_NUM || i < 0) {
                return nullptr;
            }
            return &instance[i];
        };

        SingletonQueue() = delete;

        ~SingletonQueue() = delete;

        SingletonQueue(T &&) = delete;

        SingletonQueue(const T &) = delete;

        void operator=(const T &) = delete;
    };
}
#endif