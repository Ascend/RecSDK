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

#ifndef TIMECOST_H
#define TIMECOST_H

#include <chrono>

namespace ock {
namespace ctr {
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

private:
    std::chrono::high_resolution_clock::time_point start_;
};
}
}

#endif