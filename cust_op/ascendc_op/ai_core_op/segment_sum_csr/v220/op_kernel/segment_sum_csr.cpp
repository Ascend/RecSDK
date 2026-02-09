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

namespace SegmentSumCsrKernel {
constexpr uint32_t DATA_ALIGN_BYTES = 32;
constexpr uint32_t DATA_COPY_ALIGN_BYTES = 16;
constexpr uint32_t MAX_SEGMENT_LEN = 1024;
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

template <typename csrType, typename vType>
class KernelSegmentSumCsr {
public:
    __aicore__ inline KernelSegmentSumCsr() {}
    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        this->batchSize = tilingData.batchSize;

        // 当前核的index
        this->blockIdx = AscendC::GetBlockIdx();
        // 判断当前核是前核还是尾核
        this->isTailCore = (this->blockIdx >= tilingData.remainedSegments);
        this->currentCoreSegments = isTailCore ? tilingData.baseCoreSegments : tilingData.formerCoreSegments;
        this->currentSegment = isTailCore ? tilingData.remainedSegments * tilingData.formerCoreSegments +
                                                (blockIdx - tilingData.remainedSegments) * tilingData.baseCoreSegments
                                          : blockIdx * tilingData.formerCoreSegments;

        csrSegGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ csrType*>(args.csrSeg), tilingData.csrSegLength);
        valuesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(args.values), tilingData.totalLength);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(args.y), tilingData.segmentNums);

        pipe.InitBuffer(outQueue, 1, sizeof(vType) * tilingData.segmentNums);
        pipe.InitBuffer(reducesumTmpBuf, sizeof(vType) * MAX_SEGMENT_LEN);
        pipe.InitBuffer(valuesBuf, sizeof(vType) * tilingData.totalLength);
    }
    __aicore__ inline void Process()
    {
        if (currentCoreSegments == 0U) { return; }
        Compute();
        CopyOut();
    }

    template <typename dType>
    __aicore__ inline void CpGm2Local(const AscendC::LocalTensor<dType>& lt, const AscendC::GlobalTensor<dType>& gt,
                                      int64_t len)
    {
        uint32_t alignLen = len * sizeof(dType) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(dType) - alignLen;

        AscendC::GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(dType) / 2);
        AscendC::LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        if (alignLen != 0) {
            DataCopy(uint16Lt, uint16Gt, alignLen / 2);
        }

        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadGm2Local(uint16Lt[alignLen / 2], uint16Gt[alignLen / 2], unAlignLen / 2);
#else
            const AscendC::DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint16Lt[alignLen / 2], uint16Gt[alignLen / 2], dataCopyExtParams, dataCopyPadExtParams);
#endif
        }
    }

    __aicore__ inline void DataCopyPadGm2Local(const AscendC::LocalTensor<uint16_t>& lt,
                                               const AscendC::GlobalTensor<uint16_t>& gt, int64_t len)
    {
        AscendC::DataCopy<uint16_t>(lt, gt, DATA_COPY_ALIGN_BYTES);
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        AscendC::Duplicate<uint16_t>(lt, 0, mask, 1, 1, 1);
    }

    template <typename dType>
    __aicore__ inline void CpLocal2Gm(const AscendC::GlobalTensor<dType>& gt, const AscendC::LocalTensor<dType>& lt,
                                      int64_t len)
    {
        uint32_t alignLen = len * sizeof(dType) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(dType) - alignLen;

        AscendC::GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(dType) / 2);
        AscendC::LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        if (alignLen != 0) {
            DataCopy(uint16Gt, uint16Lt, alignLen / 2);
        }
        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadLocal2Gm(uint16Gt[alignLen / 2], uint16Lt[alignLen / 2], unAlignLen / 2);
#else
            const AscendC::DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const AscendC::DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint16Gt[alignLen / 2], uint16Lt[alignLen / 2], dataCopyExtParams);
#endif
        }
    }

    __aicore__ inline void DataCopyPadLocal2Gm(const AscendC::GlobalTensor<uint16_t>& gt,
                                               const AscendC::LocalTensor<uint16_t>& lt, int64_t len)
    {
        AscendC::SetAtomicAdd<uint16_t>();
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        AscendC::Duplicate<uint16_t>(lt, 0, mask, 1, 1, 1);
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(gt, lt, DATA_COPY_ALIGN_BYTES);
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void WholeReduceSumImpl(const AscendC::LocalTensor<vType>& dst,
                                              const AscendC::LocalTensor<vType>& src,
                                              const csrType hLength,
                                              int32_t idx)
    {
        static constexpr uint32_t ONE_REPEAT_SIZE = REPEAT_LEN / sizeof(vType);
        AscendC::SetMaskCount();
        csrType totalNum = hLength;
        AscendC::LocalTensor<vType> tempTensor = reducesumTmpBuf.Get<vType>();

        AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(0, totalNum);
        AscendC::WholeReduceSum<vType, false>(tempTensor, src, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                                              DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        totalNum = AscendC::DivCeil(totalNum, ONE_REPEAT_SIZE);
        while (totalNum > 1) {
            AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(0, totalNum);
            AscendC::WholeReduceSum<vType, false>(tempTensor, tempTensor, AscendC::MASK_PLACEHOLDER, 1,
                                                  DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            totalNum = AscendC::DivCeil(totalNum, ONE_REPEAT_SIZE);
        }
        dst.SetValue(idx, tempTensor.GetValue(0));

        AscendC::ResetMask();
        AscendC::SetMaskNorm();
    }

private:

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<vType> valuesLocal = valuesBuf.Get<vType>();
        AscendC::LocalTensor<vType> outLocal = outQueue.AllocTensor<vType>();

        for (int32_t i = 0; i < this->currentCoreSegments; ++i) {
            // 此时当前核需要处理的数据所在的起始位置
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

    int64_t currentCoreSegments, currentSegment, batchSize, blockIdx;
    bool isTailCore;
};
}  // namespace SegmentSumCsrKernel

extern "C" __global__ __aicore__ void segment_sum_csr(GM_ADDR csrSeg, GM_ADDR values, GM_ADDR y,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    SegmentSumCsrKernel::Args args{csrSeg, values, y, workspace, tiling};
    if (TILING_KEY_IS(0)) {
        SegmentSumCsrKernel::KernelSegmentSumCsr<int32_t, float> op;
        op.Init(args);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        SegmentSumCsrKernel::KernelSegmentSumCsr<int64_t, float> op;
        op.Init(args);
        op.Process();
    }
}