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
==============================================================================*/

#include "kernel_operator.h"
#include "kernel_common_utils.h"

namespace SegmentSumCsrKernel {
constexpr uint32_t MAX_SEGMENT_LEN = 1024;  // 仅用于 WholeReduceSum 迭代中间结果
constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
constexpr uint32_t REPEAT_LEN = 256;
constexpr uint32_t DEFAULT_REP_STRIDE = 8;

struct Args {
    GM_ADDR csrSeg;
    GM_ADDR values;
    GM_ADDR y;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename T>
struct typeTag {};

template <typename csrType, typename vType>
class KernelSegmentSumCsr {
public:
    __aicore__ inline KernelSegmentSumCsr() {}

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        this->batchSize = tilingData.batchSize;
        this->blockIdx = AscendC::GetBlockIdx();
        this->isTailCore = (this->blockIdx >= tilingData.remainedSegments);
        this->currentCoreSegments = isTailCore ? tilingData.baseCoreSegments : tilingData.formerCoreSegments;
        this->currentSegment = isTailCore ? tilingData.remainedSegments * tilingData.formerCoreSegments +
                                                (blockIdx - tilingData.remainedSegments) * tilingData.baseCoreSegments
                                          : blockIdx * tilingData.formerCoreSegments;

        if (tilingData.segmentNums == 0) {
            this->currentCoreSegments = 0;
            return;
        }

        csrSegGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ csrType*>(args.csrSeg), tilingData.csrSegLength);
        valuesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(args.values), tilingData.totalLength);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(args.y), tilingData.segmentNums);

        const int64_t maxSegBytes = sizeof(vType) * tilingData.maxSegmentLen;
        const int64_t maxSegFloatBytes = sizeof(float) * tilingData.maxSegmentLen;

        pipe.InitBuffer(outQueue, 1, sizeof(vType) * tilingData.segmentNums);
        pipe.InitBuffer(reducesumTmpBuf, sizeof(float) * MAX_SEGMENT_LEN);
        pipe.InitBuffer(valuesBuf, maxSegBytes);
        pipe.InitBuffer(castTmpBuf, maxSegFloatBytes);
    }

    __aicore__ inline void Process()
    {
        if (currentCoreSegments == 0U) {
            return;
        }
        Compute();
        CopyOut();
    }

    __aicore__ inline AscendC::LocalTensor<float> ReduceFloat(const AscendC::LocalTensor<float>& floatSrc,
                                                              csrType hLength)
    {
        constexpr uint32_t FLOAT_ONE_REPEAT_SIZE = REPEAT_LEN / sizeof(float);
        AscendC::SetMaskCount();
        csrType totalNum = hLength;
        AscendC::LocalTensor<float> floatTemp = reducesumTmpBuf.Get<float>();

        AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(0, totalNum);
        AscendC::WholeReduceSum<float, false>(floatTemp, floatSrc, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                                              DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        totalNum = AscendC::DivCeil(totalNum, FLOAT_ONE_REPEAT_SIZE);
        while (totalNum > 1) {
            AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(0, totalNum);
            AscendC::WholeReduceSum<float, false>(floatTemp, floatTemp, AscendC::MASK_PLACEHOLDER, 1,
                                                  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            totalNum = AscendC::DivCeil(totalNum, FLOAT_ONE_REPEAT_SIZE);
        }
        return floatTemp;
    }

    __aicore__ inline void WholeReduceSumImpl(const AscendC::LocalTensor<vType>& dst,
                                              const AscendC::LocalTensor<vType>& src, const csrType hLength,
                                              int32_t idx)
    {
        WholeReduceSumImplDispatch(dst, idx, src, hLength, typeTag<vType>{});
    }

    template <typename T>
    __aicore__ inline void WholeReduceSumImplDispatch(const AscendC::LocalTensor<T>& dst, int32_t dstIdx,
                                                      const AscendC::LocalTensor<T>& src, csrType hLength, typeTag<T>)
    {
        static constexpr uint32_t ONE_REPEAT_SIZE = REPEAT_LEN / sizeof(T);
        AscendC::SetMaskCount();
        csrType totalNum = hLength;
        AscendC::LocalTensor<T> tempTensor = reducesumTmpBuf.Get<T>();

        AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(0, totalNum);
        AscendC::WholeReduceSum<T, false>(tempTensor, src, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                                          DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        totalNum = AscendC::DivCeil(totalNum, ONE_REPEAT_SIZE);
        while (totalNum > 1) {
            AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(0, totalNum);
            AscendC::WholeReduceSum<T, false>(tempTensor, tempTensor, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                                              DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            totalNum = AscendC::DivCeil(totalNum, ONE_REPEAT_SIZE);
        }
        dst.SetValue(dstIdx, tempTensor.GetValue(0));

        AscendC::ResetMask();
        AscendC::SetMaskNorm();
    }

    __aicore__ inline void WholeReduceSumImplDispatch(const AscendC::LocalTensor<half>& dst, int32_t dstIdx,
                                                      const AscendC::LocalTensor<half>& src, csrType hLength,
                                                      typeTag<half>)
    {
        uint32_t len = static_cast<uint32_t>(hLength);
        AscendC::LocalTensor<float> floatSrc = castTmpBuf.Get<float>();
        AscendC::Cast(floatSrc, src, AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> floatResult = ReduceFloat(floatSrc, hLength);

        AscendC::LocalTensor<half> halfTemp = castTmpBuf.Get<half>();
        AscendC::Cast(halfTemp, floatResult, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::PipeBarrier<PIPE_V>();
        dst.SetValue(dstIdx, halfTemp.GetValue(0));

        AscendC::ResetMask();
        AscendC::SetMaskNorm();
    }

    __aicore__ inline void WholeReduceSumImplDispatch(const AscendC::LocalTensor<bfloat16_t>& dst, int32_t dstIdx,
                                                      const AscendC::LocalTensor<bfloat16_t>& src, csrType hLength,
                                                      typeTag<bfloat16_t>)
    {
        uint32_t len = static_cast<uint32_t>(hLength);
        AscendC::LocalTensor<float> floatSrc = castTmpBuf.Get<float>();
        AscendC::Cast(floatSrc, src, AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> floatResult = ReduceFloat(floatSrc, hLength);
        float result = floatResult.GetValue(0);

        union {
            float f;
            uint32_t u;
        } conv;
        conv.f = result;
        uint16_t bf16Bits = static_cast<uint16_t>(conv.u >> 16);

        AscendC::LocalTensor<uint16_t> dstU16 = dst.template ReinterpretCast<uint16_t>();
        dstU16.SetValue(dstIdx, bf16Bits);

        AscendC::ResetMask();
        AscendC::SetMaskNorm();
    }

private:
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<vType> valuesLocal = valuesBuf.Get<vType>();
        AscendC::LocalTensor<vType> outLocal = outQueue.AllocTensor<vType>();

        for (int32_t i = 0; i < this->currentCoreSegments; ++i) {
            csrType startLoc = csrSegGlobal.GetValue(this->currentSegment + i) * batchSize;
            csrType endLoc = csrSegGlobal.GetValue(this->currentSegment + i + 1) * batchSize;
            CpGm2Local(valuesLocal, valuesGlobal[startLoc], endLoc - startLoc);
            AscendC::PipeBarrier<PIPE_ALL>();
            WholeReduceSumImpl(outLocal, valuesLocal, endLoc - startLoc, i);
        }
        outQueue.EnQue<vType>(outLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<vType> outLocal = outQueue.DeQue<vType>();
        CpLocal2Gm(yGlobal[this->currentSegment], outLocal, this->currentCoreSegments);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::GlobalTensor<csrType> csrSegGlobal;
    AscendC::GlobalTensor<vType> valuesGlobal;
    AscendC::GlobalTensor<vType> yGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reducesumTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valuesBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castTmpBuf;

    int64_t currentCoreSegments, currentSegment, batchSize, blockIdx;
    bool isTailCore;
};
}  // namespace SegmentSumCsrKernel

template <typename csrT, typename vT>
__aicore__ inline void dispatchKernel(SegmentSumCsrKernel::Args& args)
{
    SegmentSumCsrKernel::KernelSegmentSumCsr<csrT, vT> op;
    op.Init(args);
    op.Process();
}

extern "C" __global__ __aicore__ void segment_sum_csr(GM_ADDR csrSeg, GM_ADDR values, GM_ADDR y, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    SegmentSumCsrKernel::Args args{csrSeg, values, y, workspace, tiling};
    if (TILING_KEY_IS(0)) {
        dispatchKernel<int32_t, float>(args);
    } else if (TILING_KEY_IS(1)) {
        dispatchKernel<int32_t, half>(args);
    } else if (TILING_KEY_IS(2)) {
        dispatchKernel<int32_t, bfloat16_t>(args);
    } else if (TILING_KEY_IS(4)) {
        dispatchKernel<int64_t, float>(args);
    } else if (TILING_KEY_IS(5)) {
        dispatchKernel<int64_t, half>(args);
    } else if (TILING_KEY_IS(6)) {
        dispatchKernel<int64_t, bfloat16_t>(args);
    }
}