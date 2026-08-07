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

/**
 * @file block_epilogue_trans_out.hpp
 * @brief HSTU Epilogue 输出转置 Block 实现
 * @description 提供数据从 UB 转存到 GM 的操作，支持 UB_TO_GM 和 GM_TO_GM 两种转置模式，
 *              用于将计算结果从 Unified Buffer 写回到 Global Memory
 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Block {

/**
 * @brief 转置类型枚举
 * @description UB_TO_GM: 从 UB 写回 GM; GM_TO_GM: GM 到 GM 的数据搬运
 */
enum class TransTag : uint8_t {
    UB_TO_GM = 0,
    GM_TO_GM
};

/**
 * @brief Epilogue 输出转置 Block
 * @tparam ArchTag_ 架构标签
 * @tparam TileBuffer_ Tile 缓冲区类型
 * @tparam TransTag_ 转置类型
 * @tparam ElementAccumulator_ 累加器数据类型
 * @tparam Element_ 目标数据类型
 * @tparam HAS_RAB 是否有相对位置偏置
 * @description 执行计算结果从 UB 到 GM 的转存操作
 */
template <class ArchTag_, class TileBuffer_, TransTag TransTag_, class ElementAccumulator_,
          class Element_ = ElementAccumulator_, bool HAS_RAB = false>
struct BlockEpilogueTransOut {
    static_assert(DEPENDENT_FALSE<ArchTag_>, "BlockEpilogueTransOut specialization failed!.");
};

template <class ArchTag_, class TileBuffer_, class ElementAccumulator_, class Element_, bool HAS_RAB>
struct BlockEpilogueTransOut<ArchTag_, TileBuffer_, TransTag::UB_TO_GM, ElementAccumulator_, Element_, HAS_RAB> {
public:
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using Element = Element_;
    using TileBuffer = TileBuffer_;

    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    BlockEpilogueTransOut(uint32_t cubeFlag, uint32_t stride, Arch::Resource<ArchTag>& resource)
    {
        this->cubeReady = Arch::CrossCoreFlag(cubeFlag);
        this->stride = stride;
        this->ubTransOut = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::TRANS_OUT);
    }

    template <class TensorA>
    CATLASS_DEVICE void operator()(TensorA& dstTensor)
    {
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(cubeReady.id);

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto rows = tla::get<0>(dstTensor.shape());
        auto cols = tla::get<1>(dstTensor.shape());

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = rows;
        intriParams.blockLen = cols / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = (stride - cols) / ELE_NUM_PER_C0;

        AscendC::DataCopy(dstTensor.data()[dstOffset], ubTransOut, intriParams);
    }

private:
    int64_t stride{0};
    Arch::CrossCoreFlag cubeReady;

    AscendC::LocalTensor<Element> ubTransOut;
};

template <class ArchTag_, class TileBuffer_, class ElementAccumulator_, class Element_, bool HAS_RAB>
struct BlockEpilogueTransOut<ArchTag_, TileBuffer_, TransTag::GM_TO_GM, ElementAccumulator_, Element_, HAS_RAB> {
public:
    using ArchTag = ArchTag_;
    using ElementAccumulator = ElementAccumulator_;
    using Element = Element_;
    using TileBuffer = TileBuffer_;

    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM = TileBuffer::TILE_Q_GRAD_MAX_ELEM;
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    BlockEpilogueTransOut(Arch::Resource<ArchTag>& resource)
    {
        for (auto i = 0; i < STAGES; ++i) {
            ubTransIn[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::TRANS_IN[i]);
            ubTransOut[i] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::TRANS_OUT[i]);
        }
    }

    template <class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(TensorA& dstTensor, TensorB& srcTensor)
    {
        auto totalProcSeqs = tla::get<0>(srcTensor.shape());
        auto totalSeqLenQ = tla::get<0>(dstTensor.shape());
        auto dim = tla::get<1>(srcTensor.shape());
        auto head = tla::get<1>(dstTensor.shape());

        AscendC::DataCopyParams intriParams;
        intriParams.blockLen = dim / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.stride())) / ELE_NUM_PER_C0;

        uint32_t pingPongFlag = 0;
        auto eachProcSeqs = TILE_Q_GRAD_MAX_ELEM / dim;  // 每次循环处理的序列长度
        auto loopCnt = CeilDiv(totalProcSeqs, eachProcSeqs);  // 循环次数等于总序列长度 除以 每次处理的序列长度
        auto tailProcSeqs = totalProcSeqs - (loopCnt - 1) * eachProcSeqs;

        for (auto h = 0; h < head; ++h) {
            int64_t srcOffset = srcTensor.layout()(srcTensor.coord()) + h * totalSeqLenQ * dim;
            int64_t dstOffset = dstTensor.layout()(dstTensor.coord()) + h * dim;
            for (auto loopIdx = 0; loopIdx < loopCnt; ++loopIdx) {
                // 最后一段用 tailProcSeqs，其余用 eachProcSeqs
                auto procSeqs = eachProcSeqs + (loopIdx == loopCnt - 1) * (tailProcSeqs - eachProcSeqs);
                intriParams.blockCount = procSeqs;
                // copy in
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingPongFlag);
                AscendC::DataCopy(ubTransIn[pingPongFlag], srcTensor.data()[srcOffset], procSeqs * dim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingPongFlag);

                // cast
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingPongFlag);
                AscendC::Cast(ubTransOut[pingPongFlag], ubTransIn[pingPongFlag], AscendC::RoundMode::CAST_RINT,
                              procSeqs * dim);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingPongFlag);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);

                // copyout
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
                AscendC::DataCopy(dstTensor.data()[dstOffset], ubTransOut[pingPongFlag], intriParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
                pingPongFlag ^= 1;
                srcOffset += procSeqs * dim;
                dstOffset += procSeqs * tla::get<0>(dstTensor.stride());
            }
        }
    }

private:
    AscendC::LocalTensor<ElementAccumulator> ubTransIn[STAGES];
    AscendC::LocalTensor<Element> ubTransOut[STAGES];
};

}  // namespace Catlass::Epilogue::Block
