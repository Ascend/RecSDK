/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_ASYNC_TASK_H
#define EMBEDDING_CACHE_ASYNC_TASK_H
#include <future>
#include <any>
#include <functional>
#include <thread>
#include <chrono>
#include <vector>

#include "thread_pool.h"

namespace Embcache {

template <typename T>
class AsyncTask {
public:
    template <typename Func>
    explicit AsyncTask(Func&& func)
    {
        auto task_ptr = std::make_shared<std::packaged_task<T()>>(std::forward<Func>(func));
        future_ = task_ptr->get_future();
        ThreadPool::GetInstance().enqueue([task_ptr]() { (*task_ptr)(); });
    }

    T get()
    {
        return future_.get();
    }

private:
    std::future<T> future_;
};
}  // namespace Embcache

#endif  // EMBEDDING_CACHE_ASYNC_TASK_H