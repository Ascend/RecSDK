/**
 * @file relative_attn_bias_pos.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_POS_H
#define MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_POS_H
#include "rab_common.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int SEQ_EXPAND = 2;  // rab_pos中序列长度为原本输入的两倍

template <typename FloatType>
class RelativeAttnBiasPos {
public:
    __aicore__ inline RelativeAttnBiasPos() {}

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        s = SEQ_EXPAND * tilingData.s;
        bs = tilingData.bs;
        stride = tilingData.positionStride;
        for (auto i = 0; i < bs; ++i) {
            pastValidLens[i] = tilingData.pastValidLens[i];
        }

        posBiasGT.SetGlobalBuffer((__gm__ FloatType*)args.positionBias, s * s);
        identityGT.SetGlobalBuffer((__gm__ FloatType*)args.identity, s * s);
        rabPosBiasOutGT.SetGlobalBuffer((__gm__ FloatType*)args.rabPosOut, bs * s * s);

        pipe.InitBuffer(queIdentityIn, NUM_BUFFER, Ceil(SEQ_EXPAND * stride * sizeof(FloatType)));
        pipe.InitBuffer(quePosIn, NUM_BUFFER, Ceil(stride * sizeof(FloatType)));

        int64_t totalTableSizeSplit = s % GetBlockNum();
        int64_t baseLen = s / GetBlockNum();
        if (GetBlockIdx() >= totalTableSizeSplit) {
            totalRow = baseLen;
            rowOffset = totalTableSizeSplit * (baseLen + 1) + (GetBlockIdx() - totalTableSizeSplit) * baseLen;
        } else {
            totalRow = baseLen + 1;
            rowOffset = GetBlockIdx() * (baseLen + 1);
        }
        REL_POS_BIAS_FIRST = posBiasGT.GetValue(0);
    }

    __aicore__ inline void ComputeIdentity(int offset, int cnt)
    {
        // DataCopyIn identity
        LocalTensor<FloatType> identityUb = queIdentityIn.AllocTensor<FloatType>();

        DataCopy(identityUb, identityGT[offset], Ceil(cnt * sizeof(FloatType)) / sizeof(FloatType));
        queIdentityIn.EnQue(identityUb);

        // Compute identity * rel_pos_bias[0, 0], (1 - identity)
        LocalTensor<FloatType> identityFilledUb = queIdentityIn.DeQue<FloatType>();

        // 后半段 (1 - identity)
        Muls(identityFilledUb[stride], identityFilledUb, (FloatType)-1, cnt);
        Adds(identityFilledUb[stride], identityFilledUb[stride], (FloatType)1, cnt);

        // 前半段 identity * rel_pos_bias[0, 0]
        Muls(identityFilledUb, identityFilledUb, REL_POS_BIAS_FIRST, cnt);

        queIdentityIn.EnQue(identityFilledUb);
    }

    __aicore__ inline void DataCopyIn(int row, int offset, int cnt)
    {
        LocalTensor<FloatType> posBiasUb = quePosIn.AllocTensor<FloatType>();
        DataCopy(posBiasUb, posBiasGT[row * s + offset], Ceil(cnt * sizeof(FloatType)) / sizeof(FloatType));
        quePosIn.EnQue(posBiasUb);
    }

    __aicore__ inline void ComputeRabBias(LocalTensor<FloatType>& identityCalcUb, int cnt)
    {
        LocalTensor<FloatType> posBiasUb = quePosIn.DeQue<FloatType>();
        Mul(posBiasUb, posBiasUb, identityCalcUb[stride], cnt);
        Add(posBiasUb, posBiasUb, identityCalcUb, cnt);
        pipe_barrier(PIPE_ALL);
        quePosIn.EnQue(posBiasUb);
    }

    __aicore__ inline int64_t Ceil(int64_t a, int64_t b = DATA_ALIGN_BYTES)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b * b;
    }

    __aicore__ inline void DataCopyOut(int offset, int cnt)
    {
        uint32_t datasize = cnt * sizeof(FloatType);
        uint32_t alignLen = datasize / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = datasize - alignLen;
        uint32_t alignCnt = alignLen / sizeof(FloatType);
        uint32_t unAlignCnt = unAlignLen / sizeof(FloatType);

        LocalTensor<FloatType> posBiasUb = quePosIn.DeQue<FloatType>();
        // 对齐部分拷出
        if (alignLen > 0) {
            DataCopy(rabPosBiasOutGT[offset], posBiasUb, cnt);
        }
        // 非对齐部分拷出
        if (unAlignLen > 0) {
#ifdef SUPPORT_V200
            uint64_t mask0 = (1ul << (DATA_ALIGN_BYTES / sizeof(FloatType))) - (1ul << unAlignCnt);
            uint64_t mask[2] = {mask0, 0};
            Duplicate(posBiasUb[alignCnt], (FloatType)0, mask, 1, 1, 1);
            quePosIn.EnQue(posBiasUb);
            posBiasUb = quePosIn.DeQue<FloatType>();
            SetAtomicAdd<FloatType>();
            DataCopy(rabPosBiasOutGT[offset + alignCnt], posBiasUb[alignCnt], Ceil(unAlignLen) / sizeof(FloatType));
            SetAtomicNone();
#else
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            DataCopyPad(rabPosBiasOutGT[offset + alignCnt], posBiasUb[alignCnt], dataCopyExtParams);
#endif
        }
        quePosIn.FreeTensor(posBiasUb);
    }

    __aicore__ inline void Compute(Args args)
    {
        Init(args);
        for (int row = rowOffset; row < rowOffset + totalRow; ++row) {
            int offset = 0;
            for (int j = 0; j < (s + stride - 1) / stride; ++j) {
                int remain = s - offset;
                int cnt = remain > stride ? stride : remain;
                ComputeIdentity(offset + row * s, cnt);
                LocalTensor<FloatType> identityCalcUb = queIdentityIn.DeQue<FloatType>();

                for (int b = 0; b < bs; ++b) {
                    int valid_len = pastValidLens[b];
                    int valid_row = row > valid_len ? valid_len : row;
                    DataCopyIn(valid_row, offset, cnt);
                    ComputeRabBias(identityCalcUb, cnt);
                    int padOutPtr = b * s * s + row * s + j * stride;
                    DataCopyOut(padOutPtr, cnt);
                }
                queIdentityIn.FreeTensor(identityCalcUb);
                offset += cnt;
            }
        }
    }

private:
    // shape
    int s;
    int bs;
    int stride;
    // tiling
    int rowOffset;  // identity、rel_pos_bias(s, s)的行偏移
    int totalRow;   // 需要处理的总行数

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> queIdentityIn;
    TQue<TPosition::VECIN, 1> queIdentityCalcIn;
    TQue<TPosition::VECIN, 1> quePosIn;

    GlobalTensor<FloatType> identityGT;
    GlobalTensor<FloatType> posBiasGT;
    GlobalTensor<FloatType> rabPosBiasOutGT;
    uint32_t pastValidLens[MAX_BATCH_SIZE];
    FloatType REL_POS_BIAS_FIRST;  // identity[0, 0]
};

#endif  // MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_POS_H
