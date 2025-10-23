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

#ifndef TASK_QUEUE_H
#define TASK_QUEUE_H

#include <utility>
#include <atomic>
#include <list>
#include <condition_variable>

namespace MxRec {
    namespace Common {
        template<class T>
        class TaskQueue {
        public:
            TaskQueue() = default;

            ~TaskQueue() = default;

            TaskQueue(TaskQueue const &other)
            {
                std::lock_guard<std::mutex> lk(other.mut);
                dataQueue = other.dataQueue;
            }

            TaskQueue &operator=(TaskQueue const &other)
            {
                if (this == &other) {
                    return *this;
                }
                std::lock_guard<std::mutex> lk(other.mut);
                dataQueue = other.dataQueue;
                return *this;
            }

            void Pushv(T &t)
            {
                std::lock_guard<std::mutex> lk(mut);
                dataQueue.push_back(std::move(t));
                dataCond.notify_one();
            }

            void Pushv(T &&t)
            {
                std::lock_guard<std::mutex> lk(mut);
                dataQueue.emplace_back(t);
                dataCond.notify_one();
            }

            T WaitAndPop()
            {
                std::unique_lock<std::mutex> lk(mut);
                dataCond.wait(lk, [this] {
                    if (!finished) {
                        return !dataQueue.empty();
                    } else {
                        return true;
                    }
                });
                T res;
                if (finished) {
                    return std::move(res);
                }
                res = std::move(dataQueue.front());
                dataQueue.pop_front();
                return std::move(res);
            }

            void DestroyQueue()
            {
                finished = true;
                dataCond.notify_all();
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
            std::list<T> dataQueue;
            std::condition_variable dataCond;
            bool finished = false;
        };
    }
}


#endif
