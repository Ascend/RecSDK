/**
 * @file index_select_for_rank1_backward_kernel.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_INDEX_SELECT_FOR_RANK1_BACKWARD_KERNEL_H
#define MXREC_INDEX_SELECT_FOR_RANK1_BACKWARD_KERNEL_H
#include "kernel_operator.h"

using namespace AscendC;
constexpr int DATA_ALIGN_BYTES = 32;

template <typename IndexType>
class IndexSelectForRank1BackwardKernel {
public:
    __aicore__ inline IndexSelectForRank1BackwardKernel(const GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        xDim0 = tilingData.xDim0;
        stride = tilingData.stride;
        int32_t baseLen = tilingData.baseLen;
        int32_t tailSplitIndex = tilingData.tailSplitIndex;

        if (GetBlockIdx() >= tailSplitIndex) {
            totalLen = baseLen;
            indexOffset = tailSplitIndex * (baseLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseLen;
        } else {
            totalLen = baseLen + 1;
            indexOffset = GetBlockIdx() * (baseLen + 1);
        }
    }

    __aicore__ void Init(GM_ADDR gradY, GM_ADDR index, GM_ADDR gradX)
    {
        gradYGm.SetGlobalBuffer((__gm__ float*)gradY, stride);
        indexGm.SetGlobalBuffer((__gm__ IndexType*)index, stride);
        gradXGm.SetGlobalBuffer((__gm__ float*)gradX, xDim0);

        pipe.InitBuffer(inQueGradY, 1, AlignTo32(stride * sizeof(float)));
        pipe.InitBuffer(inQueIndex, 1, AlignTo32(stride * sizeof(IndexType)));
        pipe.InitBuffer(outQueGradX, 1, AlignTo32(xDim0 * sizeof(float)));

        gradYUb = inQueGradY.AllocTensor<float>();
        indexUb = inQueIndex.AllocTensor<IndexType>();
        gradXUb = outQueGradX.AllocTensor<float>();

        Duplicate(gradXUb, static_cast<float>(0), xDim0);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        gradYAddr = reinterpret_cast<__ubuf__ float*>(gradYUb.GetPhyAddr());
        indexAddr = reinterpret_cast<__ubuf__ IndexType*>(indexUb.GetPhyAddr());
        gradXAddr = reinterpret_cast<__ubuf__ float*>(gradXUb.GetPhyAddr());
    }

    __aicore__ void Process()
    {
        int32_t remain = totalLen;
        int32_t offset = indexOffset;
        while (remain > 0) {
            int32_t cnt = remain > stride ? stride : remain;
            CopyIn(offset, cnt);
            Compute(cnt);

            remain -= cnt;
            offset += cnt;
        }
        CopyOut();
    }

    template <typename T>
    __aicore__ inline void CpPadGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        if (alignLen != 0) {
            DataCopy(lt, gt, alignLen / sizeof(T));
        }
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
        }
    }

    __aicore__ void CopyIn(const int offset, int32_t cnt)
    {
        CpPadGm2Local(indexUb, indexGm[offset], cnt);
        CpPadGm2Local(gradYUb, gradYGm[offset], cnt);
    }

    __aicore__ void Compute(const int32_t cnt)
    {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        for (int i = 0; i < cnt; i++) {
            const auto index = indexAddr[i];
            const auto gradY = gradYAddr[i];
            gradXAddr[index] += gradY;
        }
    }

    __aicore__ void CopyOut()
    {
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

        SetAtomicAdd<float>();
        DataCopy(gradXGm, gradXUb, AlignTo32(xDim0 * sizeof(float)) / sizeof(float));
        SetAtomicNone();

        inQueGradY.FreeTensor<float>(gradYUb);
        inQueIndex.FreeTensor<IndexType>(indexUb);
        outQueGradX.FreeTensor<float>(gradXUb);
    }

private:
    LocalTensor<float> gradXUb;
    LocalTensor<float> gradYUb;
    LocalTensor<IndexType> indexUb;

    __ubuf__ float* gradXAddr;
    __ubuf__ float* gradYAddr;
    __ubuf__ IndexType* indexAddr;

    int32_t xDim0;

    int32_t totalLen;
    int32_t stride;
    int32_t indexOffset;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueIndex;
    TQue<TPosition::VECIN, 1> inQueGradY;
    TQue<TPosition::VECOUT, 1> outQueGradX;

    GlobalTensor<float> gradXGm, gradYGm;
    GlobalTensor<IndexType> indexGm;
};

#endif  // MXREC_INDEX_SELECT_FOR_RANK1_BACKWARD_KERNEL_H
