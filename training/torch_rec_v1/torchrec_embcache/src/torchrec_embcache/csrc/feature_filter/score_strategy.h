/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SCORE_STRATEGY_H
#define SCORE_STRATEGY_H

#include <cstdint>
#include <cmath>
#include <limits>
#include "common/common.h"

namespace Embcache {

// 比较函数用于升序排序，score1 < score2 返回true，score相等时，按key升序排序
inline bool CompareShowClickEvictScore(const std::pair<int64_t, double>& a, const std::pair<int64_t, double>& b)
{
    // score相等时，按key升序排序，id小的先淘汰
    if (std::fabs(a.second - b.second) < std::numeric_limits<double>::epsilon()) {
        return a.first < b.first;  // 分数相等时，按key升序排序
    }
    // return IsFloatLess(a.second, b.second);  // 分数不等时，按分数升序排序
    return a.second < b.second;  // 分数不等时，按分数升序排序
}

// 计算准入分数 score = alpha * showCount + beta * clickCount
inline double ComputeShowClickAdmitScore(int64_t count, int64_t click, const ShowClickParams& config)
{
    return config.alpha * count + config.beta * click;
}

// 计算淘汰分数 score = (oldScore + alpha * count + beta * click) * scoreDecay
inline double ComputeShowClickEvictScore(double oldScore, int64_t count, int64_t click, const ShowClickParams& config)
{
    double new_score = oldScore + config.alpha * count + config.beta * click;
    new_score = new_score * config.scoreDecay;
    return new_score;
}

}  // namespace Embcache

#endif  // SCORE_STRATEGY_H
