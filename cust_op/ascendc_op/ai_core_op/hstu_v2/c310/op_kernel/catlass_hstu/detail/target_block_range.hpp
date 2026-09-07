/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

#pragma once

namespace Catlass::Detail {

template <uint32_t Q_BLOCK_SIZE, uint32_t K_BLOCK_SIZE>
CATLASS_DEVICE uint32_t GetTargetQBlockEnd(uint32_t kBlockId, uint32_t seqlenQ, uint32_t seqlenK, uint32_t numTarget,
                                           uint32_t numContext, uint32_t targetGroupSize)
{
    const uint32_t totalQBlocks = CeilDiv(seqlenQ, Q_BLOCK_SIZE);
    if (numTarget == 0 || targetGroupSize == 0) {
        return totalQBlocks;
    }

    const int64_t historyLen = static_cast<int64_t>(seqlenK) - numTarget;
    const int64_t kBlockBegin = static_cast<int64_t>(kBlockId) * K_BLOCK_SIZE;
    if (kBlockBegin < historyLen) {
        return totalQBlocks;
    }

    const int64_t kBlockEnd = (static_cast<int64_t>(kBlockId) + 1) * K_BLOCK_SIZE;
    const int64_t qkOffset = static_cast<int64_t>(seqlenK) - seqlenQ;
    const int64_t groupSize = targetGroupSize;
    const int64_t targetIndex = (kBlockEnd - historyLen + groupSize - 1) / groupSize;

    int64_t qElementEnd = kBlockEnd - qkOffset;
    if (static_cast<int64_t>(numContext) > qElementEnd) {
        qElementEnd = numContext;
    }
    const int64_t targetQEnd = historyLen - qkOffset + targetIndex * groupSize;
    if (targetQEnd > qElementEnd) {
        qElementEnd = targetQEnd;
    }
    if (qElementEnd > static_cast<int64_t>(seqlenQ)) {
        qElementEnd = seqlenQ;
    }
    if (qElementEnd <= 0) {
        return 0;
    }

    return static_cast<uint32_t>((qElementEnd + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE);
}

}  // namespace Catlass::Detail
