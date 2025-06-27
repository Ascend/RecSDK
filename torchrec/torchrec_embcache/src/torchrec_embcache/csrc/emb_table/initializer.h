/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H
#define EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H

#include <random>
#include <algorithm>

namespace Embcache {

class Initializer {
public:
    static void GenUniform(float* array, size_t size, float minVal, float maxVal)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(minVal, maxVal);
        std::generate(array, array + size, [&]() { return distrib(gen); });
    }

    static void GenLinear(float* array, size_t size, float minVal, float maxVal)
    {
        if (size == 0) {
            return;
        }
        if (size == 1) {
            array[0] = minVal;
            return;
        }
        for (size_t i = 0; i < size; ++i) {
            array[i] = minVal + 1.0 * i / (size - 1) * (maxVal - minVal);
        }
    }
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H
