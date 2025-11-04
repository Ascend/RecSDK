/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H
#define EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H

#include <c10/util/flat_hash_map.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>

#include "securec.h"
#include "common/common.h"
#include "utils/logger.h"

using RandomVPool = std::vector<std::vector<float>>;

namespace Embcache {

struct WeightInitParam {
    float mean;
    float stddev;
    float minVal;
    float maxVal;
};

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

    static void GenTruncatedNormal(float* array, size_t size, WeightInitParam weightParam,
                                   unsigned int seed = std::random_device{}())
    {
        if (array == nullptr || size == 0 || weightParam.stddev <= 0.0f || weightParam.minVal >= weightParam.maxVal) {
            return;
        }

        std::mt19937 gen(seed);
        std::normal_distribution<float> distrib(weightParam.mean, weightParam.stddev);

        std::generate(array, array + size, [&]() {
            float val = distrib(gen);
            while (val < weightParam.minVal || val > weightParam.maxVal) {
                val = distrib(gen);
            }
            return val;
        });
    }

    static void InitEmbeddingWeights(float* embeddingAddr, const EmbConfig& cfg)
    {
        if (cfg.initializerType == InitializerType::LINEAR) {
            Initializer::GenLinear(embeddingAddr, cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
        } else if (cfg.initializerType == InitializerType::TRUNCATED_NORMAL) {
            WeightInitParam param = {cfg.weightInitMean, cfg.weightInitStddev, cfg.weightInitMin, cfg.weightInitMax};
            Initializer::GenTruncatedNormal(embeddingAddr, cfg.embDim, param);
        } else {
            Initializer::GenUniform(embeddingAddr, cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
        }
    }

    static void InitEmbeddingWeightsLimitPool(float* embeddingAddr, const EmbConfig& cfg)
    {
        static ska::flat_hash_map<int32_t, RandomVPool> staticPoolMap;
        static std::default_random_engine engine;
        if (staticPoolMap.find(cfg.embDim) == staticPoolMap.end()) {
            engine.seed(abs(cfg.seed));
            RandomVPool staticPool =
                std::vector<std::vector<float>>(cfg.initializerRandomPoolSize, std::vector<float>(cfg.embDim));
            for (int i = 0; i < cfg.initializerRandomPoolSize; i++) {
                if (cfg.initializerType == InitializerType::LINEAR) {
                    Initializer::GenLinear(staticPool[i].data(), cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
                } else if (cfg.initializerType == InitializerType::TRUNCATED_NORMAL) {
                    WeightInitParam param = {cfg.weightInitMean, cfg.weightInitStddev, cfg.weightInitMin,
                                             cfg.weightInitMax};
                    Initializer::GenTruncatedNormal(staticPool[i].data(), cfg.embDim, param);
                } else {
                    Initializer::GenUniform(staticPool[i].data(), cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
                }
            }
            staticPoolMap.emplace(cfg.embDim, staticPool);
        }
        RandomVPool& staticPool = staticPoolMap.find(cfg.embDim)->second;
        std::uniform_int_distribution<int> uDistribution(0, cfg.initializerRandomPoolSize - 1);
        int randIndex = uDistribution(engine);
        if (embeddingAddr == nullptr) {
            LOG_ERROR("embeddingAddr is nullptr");
            throw std::runtime_error("embeddingAddr is nullptr");
        }
        auto ret = memcpy_s(embeddingAddr, cfg.embDim * sizeof(float), staticPool[randIndex].data(),
                            cfg.embDim * sizeof(float));
        if (ret != 0) {
            LOG_ERROR("memcpy_s failed when init optimizer data.");
            throw std::runtime_error("memcpy_s failed when init optimizer data.");
        }
    }
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H
