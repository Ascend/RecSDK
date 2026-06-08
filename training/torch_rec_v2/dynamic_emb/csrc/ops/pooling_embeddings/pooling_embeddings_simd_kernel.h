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
==============================================================================*/
#pragma once

#include <cstdint>
#include <type_traits>

#include "../ops_utils.h"
#include "kernel_operator.h"
#include "pooling_embeddings_simd_tiling.h"

using namespace AscendC;

namespace dyn_emb_pooling_embeddings_simd {

template <typename OffsetT, typename SrcT, typename DstT>
class PoolingEmbeddingsSimd {
public:
    static constexpr uint32_t kFloatElemsPer32B = 8U;
    // UB 中 float 槽位数：accumUb（累加）+ rowUb（单行嵌入）
    static constexpr uint32_t kFloatUbSlotCount = 2U;
    static constexpr uint32_t kRowUbSlotIndex = 1U;
    // offset 长度为 numVec + 1，末元素用于划分最后一个 pool 的 end
    static constexpr uint32_t kOffsetLengthExtra = 1U;
    static constexpr bool kSrcIsFloat = std::is_same_v<SrcT, float>;
    static constexpr bool kDstIsFloat = std::is_same_v<DstT, float>;
    static constexpr bool kNeedStaging = !kSrcIsFloat || !kDstIsFloat;

    __aicore__ inline static uint32_t EvSizeAlignedElemCount(uint32_t evSize)
    {
        return ((evSize + kFloatElemsPer32B - 1U) / kFloatElemsPer32B) * kFloatElemsPer32B;
    }

    __aicore__ inline static bool IsEvRowBytes32BAligned(uint32_t evSize, uint32_t elemBytes)
    {
        return (evSize * elemBytes) % static_cast<uint32_t>(kFloatElemsPer32B * sizeof(float)) == 0U;
    }

    __aicore__ inline explicit PoolingEmbeddingsSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR offsetData, GM_ADDR inverseData,
                                const __gm__ PoolingEmbeddingsSimdTilingData* tiling)
    {
        tiling_ = tiling;
        const uint32_t evSize = static_cast<uint32_t>(tiling_->evSize);
        const uint32_t evSizeAligned = EvSizeAlignedElemCount(evSize);
        uint64_t ubBytes = static_cast<uint64_t>(evSizeAligned) * sizeof(float) * kFloatUbSlotCount;
        if constexpr (kNeedStaging) {
            const uint32_t stagingElemBytes =
                static_cast<uint32_t>((sizeof(SrcT) > sizeof(DstT)) ? sizeof(SrcT) : sizeof(DstT));
            ubBytes += static_cast<uint64_t>(evSizeAligned) * static_cast<uint64_t>(stagingElemBytes);
        }
        pipe_->InitBuffer(ubMem_, ubBytes);

        const uint64_t srcElems = static_cast<uint64_t>(tiling_->srcNumRows) * static_cast<uint64_t>(evSize);
        const uint64_t dstElems = static_cast<uint64_t>(tiling_->batchSize) * static_cast<uint64_t>(tiling_->totalDims);
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(src), srcElems);
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dst), dstElems);
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OffsetT*>(offsetData),
                                  static_cast<uint64_t>(tiling_->numVec) + kOffsetLengthExtra);
        inverseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OffsetT*>(inverseData),
                                   static_cast<uint64_t>(tiling_->inverseLen));
        // 与 SIMT 路径一致：在设备侧用 offset[0] 作为 base，避免 Host 读取与 GM 不一致
        offsetBase_ = static_cast<int64_t>(offsetGm_.GetValue(0U));
        evSizeAligned_ = evSizeAligned;
    }

    __aicore__ inline LocalTensor<SrcT> GetStagingSrc(const LocalTensor<uint8_t>& stagingBytes) const
    {
        return stagingBytes.ReinterpretCast<SrcT>();
    }

    __aicore__ inline LocalTensor<DstT> GetStagingDst(const LocalTensor<uint8_t>& stagingBytes) const
    {
        return stagingBytes.ReinterpretCast<DstT>();
    }

    __aicore__ inline void CopyGmRowToFloatUb(const LocalTensor<float>& rowFloat, int32_t srcRowIdx, uint32_t evSize)
    {
        const uint64_t rowBase = static_cast<uint64_t>(srcRowIdx) * static_cast<uint64_t>(evSize);
        if constexpr (kSrcIsFloat) {
            if (IsEvRowBytes32BAligned(evSize, static_cast<uint32_t>(sizeof(float)))) {
                DataCopy(rowFloat, srcGm_[rowBase], evSizeAligned_);
                return;
            }
            const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(evSize * sizeof(float)), 0U, 0U, 0U};
            const DataCopyPadExtParams<float> padParams{true, 0U, 0U, 0U};
            DataCopyPad(rowFloat, srcGm_[rowBase], copyParams, padParams);
            return;
        }

        LocalTensor<float> ub = ubMem_.Get<float>();
        LocalTensor<uint8_t> stagingBytes = ub[evSizeAligned_ * kFloatUbSlotCount].ReinterpretCast<uint8_t>();
        LocalTensor<SrcT> staging = GetStagingSrc(stagingBytes);
        if (IsEvRowBytes32BAligned(evSize, static_cast<uint32_t>(sizeof(SrcT)))) {
            DataCopy(staging, srcGm_[rowBase], evSizeAligned_);
        } else {
            const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(evSize * sizeof(SrcT)), 0U, 0U, 0U};
            const DataCopyPadExtParams<SrcT> padParams{true, 0U, 0U, 0U};
            DataCopyPad(staging, srcGm_[rowBase], copyParams, padParams);
        }
        ops_utils::SyncMte2V();
        Cast(rowFloat, staging, RoundMode::CAST_NONE, evSize);
        ops_utils::SyncVMte3();
    }

    __aicore__ inline void CopyFloatUbToDst(int32_t dstOffset, const LocalTensor<float>& accumFloat, uint32_t evSize)
    {
        if constexpr (kDstIsFloat) {
            if (IsEvRowBytes32BAligned(evSize, static_cast<uint32_t>(sizeof(float)))) {
                DataCopy(dstGm_[static_cast<uint64_t>(dstOffset)], accumFloat, evSizeAligned_);
                return;
            }
            const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(evSize * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(dstGm_[static_cast<uint64_t>(dstOffset)], accumFloat, copyParams);
            return;
        }

        LocalTensor<float> ub = ubMem_.Get<float>();
        LocalTensor<uint8_t> stagingBytes = ub[evSizeAligned_ * kFloatUbSlotCount].ReinterpretCast<uint8_t>();
        LocalTensor<DstT> staging = GetStagingDst(stagingBytes);
        Cast(staging, accumFloat, RoundMode::CAST_RINT, evSize);
        ops_utils::SyncVMte3();
        if (IsEvRowBytes32BAligned(evSize, static_cast<uint32_t>(sizeof(DstT)))) {
            DataCopy(dstGm_[static_cast<uint64_t>(dstOffset)], staging, evSizeAligned_);
        } else {
            const DataCopyExtParams copyParams{1U, static_cast<uint32_t>(evSize * sizeof(DstT)), 0U, 0U, 0U};
            DataCopyPad(dstGm_[static_cast<uint64_t>(dstOffset)], staging, copyParams);
        }
    }

    __aicore__ inline void ProcessOneGroup(int32_t indicesIndex)
    {
        const uint32_t evSize = static_cast<uint32_t>(tiling_->evSize);
        const int32_t batchSize = tiling_->batchSize;
        const int32_t totalDims = tiling_->totalDims;
        const int32_t accumDims = tiling_->accumDims;

        const int64_t start =
            static_cast<int64_t>(offsetGm_.GetValue(static_cast<uint32_t>(indicesIndex))) - offsetBase_;
        const int64_t vectorNum =
            static_cast<int64_t>(offsetGm_.GetValue(static_cast<uint32_t>(indicesIndex) + kOffsetLengthExtra)) -
            static_cast<int64_t>(offsetGm_.GetValue(static_cast<uint32_t>(indicesIndex)));

        const int32_t dstRowIndex = indicesIndex % batchSize;
        const int32_t dstColIndex = indicesIndex / batchSize;
        const int32_t dstOffset = dstRowIndex * totalDims + accumDims + dstColIndex * static_cast<int32_t>(evSize);

        LocalTensor<float> ub = ubMem_.Get<float>();
        LocalTensor<float> accumUb = ub[0];
        LocalTensor<float> rowUb = ub[evSizeAligned_ * kRowUbSlotIndex];
        Duplicate<float>(accumUb, 0.0f, evSizeAligned_);

        if (vectorNum > 0) {
            // Duplicate 走 Vector；后续 Copy 走 MTE2，且 Add 会读 accumUb —— 先同步再进循环
            ops_utils::SyncVMte3();
            ops_utils::SyncMte3Mte2();
            for (int64_t j = 0; j < vectorNum; ++j) {
                const int32_t srcRowIdx = static_cast<int32_t>(inverseGm_.GetValue(static_cast<uint32_t>(start + j)));
                CopyGmRowToFloatUb(rowUb, srcRowIdx, evSize);
                ops_utils::SyncMte2V();
                Add<float>(accumUb, accumUb, rowUb, evSize);
                // Add 读 rowUb；下一轮 Copy 会覆写 rowUb —— 必须等 Vector 结束再启动 MTE2
                ops_utils::SyncVMte3();
                ops_utils::SyncMte3Mte2();
            }
            if (tiling_->combiner > 0) {
                const float invCount = 1.0f / static_cast<float>(vectorNum);
                Muls<float>(accumUb, accumUb, invCount, evSize);
            }
        }

        ops_utils::SyncVMte3();
        CopyFloatUbToDst(dstOffset, accumUb, evSize);
        ops_utils::SyncMte3Mte2();
    }

    __aicore__ inline void Process()
    {
        const int32_t coreId = static_cast<int32_t>(GetBlockIdx());
        if (coreId >= tiling_->needCoreNum) {
            return;
        }

        const int32_t numVec = tiling_->numVec;
        const int32_t stride = tiling_->needCoreNum;
        for (int32_t indicesIndex = coreId; indicesIndex < numVec; indicesIndex += stride) {
            ProcessOneGroup(indicesIndex);
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> ubMem_;
    GlobalTensor<SrcT> srcGm_;
    GlobalTensor<DstT> dstGm_;
    GlobalTensor<OffsetT> offsetGm_;
    GlobalTensor<OffsetT> inverseGm_;
    int64_t offsetBase_{0};
    uint32_t evSizeAligned_{0};
    const __gm__ PoolingEmbeddingsSimdTilingData* tiling_;
};

}  // namespace dyn_emb_pooling_embeddings_simd
