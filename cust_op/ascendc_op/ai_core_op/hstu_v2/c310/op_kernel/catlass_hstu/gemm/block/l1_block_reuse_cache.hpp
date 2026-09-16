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

#include "catlass/catlass.hpp"

namespace Catlass::Gemm::Block {

/**
 * @brief 管理按 blockId 映射的 L1 子槽位及其驻留标签
 *
 * @tparam MAX_REUSE_BLOCKS 编译期最大 slot 数，用于确定标签数组大小
 *
 * @description 组件不拥有 L1 内存，只记录 slot -> blockId 的映射。Scheduler 提供遍历是否折返及
 *              当前遍历位置，Cache 结合自身 reuseBlockCount 和驻留标签计算最终 canReuse。
 */
template <uint32_t MAX_REUSE_BLOCKS>
class L1BlockReuseCache {
public:
    static_assert(MAX_REUSE_BLOCKS > 0, "MAX_REUSE_BLOCKS must be greater than zero");

    struct ProbeResult {
        uint32_t slot{0};
        bool canReuse{false};
    };

    CATLASS_DEVICE
    explicit L1BlockReuseCache(uint32_t reuseBlockCount)
    {
        if (reuseBlockCount == 0 || reuseBlockCount > MAX_REUSE_BLOCKS) {
            this->reuseBlockCount = 1;
        } else {
            this->reuseBlockCount = reuseBlockCount;
        }
        ResetTags();
    }

    CATLASS_DEVICE
    void UpdateContext(uint32_t batchId, uint32_t headId)
    {
        if (cachedBatchId == batchId && cachedHeadId == headId) {
            return;
        }
        ResetTags();
        cachedBatchId = batchId;
        cachedHeadId = headId;
    }

    CATLASS_DEVICE
    ProbeResult Probe(int32_t blockId, bool triggerSwizzle, uint32_t swizzlePosition) const
    {
        ProbeResult result;
        // -1 是无效标签，不能把无效 blockId 当作缓存命中。
        if (blockId < 0) {
            return result;
        }
        // 连续 block 通过取模映射到不同 slot；相隔 reuseBlockCount 的 block 会复用同一个 slot。
        result.slot = static_cast<uint32_t>(blockId) % reuseBlockCount;
        // 复用窗口由 cache 自身的容量决定；scheduler 只提供是否折返和当前遍历位置。
        bool inReuseWindow = triggerSwizzle && swizzlePosition < reuseBlockCount;
        // tag 必须命中，避免被 mask/target 跳过的 block 误用同 slot 中的旧数据。
        result.canReuse = inReuseWindow && cachedBlockIds[result.slot] == blockId;
        return result;
    }

    CATLASS_DEVICE
    void Commit(uint32_t slot, int32_t blockId)
    {
        if (blockId < 0 || slot >= reuseBlockCount || slot != static_cast<uint32_t>(blockId) % reuseBlockCount) {
            return;
        }
        cachedBlockIds[slot] = blockId;
    }

private:
    CATLASS_DEVICE
    void ResetTags()
    {
        for (uint32_t i = 0; i < MAX_REUSE_BLOCKS; ++i) {
            cachedBlockIds[i] = -1;
        }
    }

    // 每个 slot 当前实际驻留的 blockId；-1 表示无有效数据。
    int32_t cachedBlockIds[MAX_REUSE_BLOCKS];
    // 当前实例实际启用的 slot 数；Q cache 和 Grad cache 可以取不同值。
    uint32_t reuseBlockCount{1};
    uint32_t cachedBatchId{static_cast<uint32_t>(-1)};
    uint32_t cachedHeadId{static_cast<uint32_t>(-1)};
};

}  // namespace Catlass::Gemm::Block
