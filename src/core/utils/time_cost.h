/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:  time cost profile module.
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#ifndef TIMECOST_H
#define TIMECOST_H

#include <chrono>
namespace MxRec {
    class TimeCost {
    public:
        TimeCost() noexcept
        {
            start_ = std::chrono::high_resolution_clock::now();
        }

        double ElapsedSec()
        {
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> d =
                    std::chrono::duration_cast < std::chrono::duration < double >> (end - start_);
            return d.count();
        }

        size_t ElapsedMS()
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::milliseconds d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
            return d.count();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_;
    };
}
#endif