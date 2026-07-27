/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

/*!
 * \file base_info.h
 * \brief
 */

#ifndef BASE_INFO_H
#define BASE_INFO_H

#include <cstdint>
#include <vector>
#include "load_balance_common.h"
#include <unordered_map>

namespace load_balance {

/**
 * This interface provides basic shape info for Balancer
 * This interface should be implemented by Operator itself
 */
class IBaseInfo {
public:
    IBaseInfo() = default;
    virtual ~IBaseInfo() = default;

    [[nodiscard]] virtual uint32_t GetBatchSize() const = 0;
    [[nodiscard]] virtual uint32_t GetGroupSize() const = 0;
    [[nodiscard]] virtual uint32_t GetQueryHeadNum() const = 0;
    [[nodiscard]] virtual uint32_t GetKvHeadNum() const = 0;
    [[nodiscard]] virtual uint32_t GetHeadDim() const = 0;
    [[nodiscard]] virtual uint32_t GetQuerySeqSize() const = 0;
    [[nodiscard]] virtual uint32_t GetQuerySeqSize(uint32_t batchIdx) const = 0;
    [[nodiscard]] virtual uint32_t GetKvSeqSize() const = 0;
    [[nodiscard]] virtual uint32_t GetKvSeqSize(uint32_t batchIdx) const = 0;
    [[nodiscard]] virtual SparseMode GetSparseMode() const = 0;
    [[nodiscard]] virtual int64_t GetPreTokenLeftUp(uint32_t querySeq, uint32_t kvSeq) const = 0;
    [[nodiscard]] virtual int64_t GetNextTokenLeftUp(uint32_t querySeq, uint32_t kvSeq) const = 0;
    [[nodiscard]] virtual bool GetIsS1G() const = 0;
    [[nodiscard]] virtual Layout GetQueryLayout() const = 0;
    [[nodiscard]] virtual Layout GetKvLayout() const = 0;
    [[nodiscard]] virtual DataType GetQueryDataType() const = 0;
    [[nodiscard]] virtual DataType GetKvDataType() const = 0;
};

/**
 * BaseInfo represents as a standard IBaseInfo for convenience
 */
class BaseInfo : public IBaseInfo {
public:
    BaseInfo() = default;
    ~BaseInfo() override = default;

    [[nodiscard]] uint32_t GetBatchSize() const override
    {
        return batchSize;
    }

    [[nodiscard]] uint32_t GetGroupSize() const override
    {
        return SafeFloorDiv(queryHeadNum, kvHeadNum, 1U);
    };

    [[nodiscard]] uint32_t GetQueryHeadNum() const override
    {
        return queryHeadNum;
    }

    [[nodiscard]] uint32_t GetKvHeadNum() const override
    {
        return kvHeadNum;
    }

    [[nodiscard]] uint32_t GetHeadDim() const override
    {
        return headDim;
    }

    [[nodiscard]] uint32_t GetQuerySeqSize() const override
    {
        return querySeqSize;
    }

    [[nodiscard]] uint32_t GetQuerySeqSize(uint32_t batchIdx) const override
    {
        if (actualQuerySeqSize.empty()) {
            return querySeqSize;
        }

        if (actualQuerySeqSize.size() == 1U) {
            return static_cast<uint32_t>(actualQuerySeqSize[0]);
        }

        if (!isCumulativeQuerySeq) {
            return static_cast<uint32_t>(actualQuerySeqSize[batchIdx]);
        }

        return (batchIdx == 0)
                   ? static_cast<uint32_t>(actualQuerySeqSize[batchIdx])
                   : static_cast<uint32_t>(actualQuerySeqSize[batchIdx] - actualQuerySeqSize[batchIdx - 1U]);
    }

    [[nodiscard]] uint32_t GetKvSeqSize() const override
    {
        return kvSeqSize;
    }

    [[nodiscard]] uint32_t GetKvSeqSize(uint32_t batchIdx) const override
    {
        if (actualKvSeqSize.empty()) {
            return kvSeqSize;
        }

        if (actualKvSeqSize.size() == 1U) {
            return static_cast<uint32_t>(actualKvSeqSize[0]);
        }

        if (!isCumulativeKvSeq) {
            return static_cast<uint32_t>(actualKvSeqSize[batchIdx]);
        }

        return (batchIdx == 0) ? static_cast<uint32_t>(actualKvSeqSize[batchIdx])
                               : static_cast<uint32_t>(actualKvSeqSize[batchIdx] - actualKvSeqSize[batchIdx - 1U]);
    }

    [[nodiscard]] SparseMode GetSparseMode() const override
    {
        if (!attenMaskFlag) {
            return SparseMode::BUTT;
        }

        if (sparseMode > static_cast<uint32_t>(SparseMode::BUTT)) {
            return SparseMode::BUTT;
        }
        return static_cast<SparseMode>(sparseMode);
    }

    [[nodiscard]] int64_t GetPreTokenLeftUp(uint32_t querySeq, uint32_t kvSeq) const override
    {
        auto mode = GetSparseMode();
        switch (mode) {
            case SparseMode::BAND:
                return static_cast<int64_t>(querySeq) - static_cast<int64_t>(kvSeq) + preToken;
            default:
                return preToken;
        }
    }

    [[nodiscard]] int64_t GetNextTokenLeftUp(uint32_t querySeq, uint32_t kvSeq) const override
    {
        auto mode = GetSparseMode();
        switch (mode) {
            case SparseMode::DEFAULT_MASK:
            case SparseMode::ALL_MASK:
            case SparseMode::LEFT_UP_CAUSAL:
                return nextToken;
            case SparseMode::RIGHT_DOWN_CAUSAL:
                return static_cast<int64_t>(kvSeq) - static_cast<int64_t>(querySeq);
            case SparseMode::BAND:
                return static_cast<int64_t>(kvSeq) - static_cast<int64_t>(querySeq) + nextToken;
            default:
                return nextToken;
        }
    }

    [[nodiscard]] bool GetIsS1G() const override
    {
        return (layoutQuery == Layout::TND || layoutQuery == Layout::BSH || layoutQuery == Layout::BSND);
    }

    [[nodiscard]] Layout GetQueryLayout() const override
    {
        return layoutQuery;
    }

    [[nodiscard]] Layout GetKvLayout() const override
    {
        return layoutKv;
    }

    [[nodiscard]] DataType GetQueryDataType() const override
    {
        return queryType;
    }

    [[nodiscard]] DataType GetKvDataType() const override
    {
        return kvType;
    }

public:
    // Batch 维大小，即一次参与负载均衡计算的样本数量 B。
    uint32_t batchSize{0U};
    // Query 的注意力头数 Nq；与 kvHeadNum 的比值为 GQA 的分组大小 G。
    uint32_t queryHeadNum{0U};
    // Query 序列长度的统一值或最大值 S1；未提供 actualQuerySeqSize 时，所有 batch 均使用该值。
    uint32_t querySeqSize{0U};
    // Key/Value 共享的注意力头数 Nkv；MHA 中等于 queryHeadNum，GQA/MQA 中通常更小。
    uint32_t kvHeadNum{0U};
    // Key/Value 序列长度的统一值或最大值 S2；未提供 actualKvSeqSize 时，所有 batch 均使用该值。
    uint32_t kvSeqSize{0U};
    // 单个注意力头的特征维度 D（head size）。
    uint32_t headDim{64U};
    // 是否启用 attention mask；为 false 时忽略 sparseMode，并按非稀疏模式处理。
    bool attenMaskFlag{false};
    // 稀疏/掩码模式，取值对应 SparseMode 枚举，用于决定有效注意力区域。
    uint32_t sparseMode{0U};
    // Query 位置左侧（历史方向）允许参与注意力的 token 数；BAND 模式下会结合 S1、S2 修正。
    uint32_t preToken{0U};
    // Query 位置右侧（未来方向）允许参与注意力的 token 数；因果/BAND 模式下会结合 S1、S2 修正。
    uint32_t nextToken{0U};
    // actualQuerySeqSize 是否保存 Query 的累计序列末端位置；为 true 时相邻元素之差才是单 batch 长度。
    bool isCumulativeQuerySeq{false};
    // actualKvSeqSize 是否保存 Key/Value 的累计序列末端位置；为 true 时相邻元素之差才是单 batch 长度。
    bool isCumulativeKvSeq{false};
    // 各 batch 的实际 Query 序列长度或累计末端位置；为空时回退使用 querySeqSize。
    std::vector<int64_t> actualQuerySeqSize{};
    // 各 batch 的实际 Key/Value 序列长度或累计末端位置；为空时回退使用 kvSeqSize。
    std::vector<int64_t> actualKvSeqSize{};
    // Query 张量的物理/逻辑布局，取值对应 Layout 枚举（如 BSND、TND、BSH）。
    Layout layoutQuery{Layout::BSND};
    // Key/Value 张量的物理/逻辑布局，取值对应 Layout 枚举。
    Layout layoutKv{Layout::BSND};
    // Query 元素的数据类型，用于负载量、访存量等估算。
    DataType queryType{DataType::FP16};
    // Key/Value 元素的数据类型，用于负载量、访存量等估算。
    DataType kvType{DataType::FP16};
};

}  // namespace load_balance
#endif  // BASE_INFO_H
