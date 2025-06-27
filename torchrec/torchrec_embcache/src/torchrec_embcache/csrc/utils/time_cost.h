/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_UTILS_TIMECOST_H
#define EMBEDDING_CACHE_UTILS_TIMECOST_H

#include <chrono>

namespace Embcache {

class TimeCost {
public:
    TimeCost()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double ElapsedSec()
    {
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start_);
        return d.count();
    }

    size_t ElapsedMS()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::milliseconds d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        return d.count();
    }

    size_t ElapsedUS()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds d = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return d.count();
    }

    size_t ElapsedNS()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds d = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
        return d.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_UTILS_TIMECOST_H