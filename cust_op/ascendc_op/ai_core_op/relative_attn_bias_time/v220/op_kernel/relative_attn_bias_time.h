/**
 * @file relative_attn_bias_time.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TIME_H
#define MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TIME_H
#include <type_traits>
#include "rab_common.h"
#include "kernel_operator.h"
using namespace AscendC;

struct SequenceParams {
    int startIndexGT;
    int startIndexUb;
    int subValue;
};

template <typename FloatType>
class RelativeAttnBiasTime {
public:
    __aicore__ inline RelativeAttnBiasTime() {}

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        s = tilingData.s;
        bs = tilingData.bs;
        stride = tilingData.stride;
        alignSeqLen = Ceil(s * sizeof(FloatType)) / sizeof(FloatType);

        int totalLen = bs * s;
        uint32_t seqDatasize = s * sizeof(FloatType);
        alignLen = seqDatasize / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        alignCnt = alignLen / sizeof(FloatType);
        unalignLen = seqDatasize - alignLen;
        unalignCnt = unalignLen / sizeof(FloatType);

        div = 1 / tilingData.bucketDivisor;
        numBuckets = tilingData.numBuckets;
        alignNumBuckets = Ceil(numBuckets * sizeof(FloatType)) / sizeof(FloatType);
        numLayer = tilingData.numLayer;

        clampMin = 1;  // 根据仿真代码，指定为1
        clampMax = tilingData.clampMax;

        timestampsGT.SetGlobalBuffer((__gm__ int32_t*)args.timestamps, bs * s);
        timestampsWeightsGT.SetGlobalBuffer((__gm__ FloatType*)args.timestampsWeights, numBuckets * numLayer);
        rabTimeBiasOutGT.SetGlobalBuffer((__gm__ FloatType*)args.rabTimeOut, numLayer * bs * s * s);
        bucketTimestampsOutGT.SetGlobalBuffer((__gm__ int32_t*)args.bucketTimestampsOut, bs * s * s);

        pipe.InitBuffer(queTimestamps, 1, stride * alignSeqLen * sizeof(int32_t));
        pipe.InitBuffer(queTimestampsFloat, 1, stride * alignSeqLen * sizeof(float));
        pipe.InitBuffer(queTimestampsWeights, 1, alignNumBuckets * numLayer * sizeof(FloatType));
        pipe.InitBuffer(tmpQue, 1, Ceil(tilingData.buffSize));

        int totalTableSizeSplit = totalLen % GetBlockNum();
        int baseLen = totalLen / GetBlockNum();
        if (GetBlockIdx() >= totalTableSizeSplit) {
            processRowLen = baseLen;
            startIndex = totalTableSizeSplit * (baseLen + 1) + (GetBlockIdx() - totalTableSizeSplit) * baseLen;
        } else {
            processRowLen = baseLen + 1;
            startIndex = GetBlockIdx() * (baseLen + 1);
        }
    }

    __aicore__ inline void FillSeqParams(SequenceParams* params, int offset, int cnt)
    {
        LocalTensor<int32_t> ts = queTimestamps.AllocTensor<int32_t>();
        DataCopy(ts, timestampsGT[offset], Ceil(cnt));
        queTimestamps.EnQue(ts);
        ts = queTimestamps.DeQue<int32_t>();
        for (int i = 0; i < cnt; ++i) {
            int seqSubValue = ts.GetValue(i);
            int seqId = (offset + i) / s;
            int seqOffsetUb = i * alignSeqLen;
            int seqOffsetGT = seqId * s;

            params[i].startIndexGT = seqOffsetGT;
            params[i].startIndexUb = seqOffsetUb;
            params[i].subValue = seqSubValue;
        }
        queTimestamps.FreeTensor(ts);
    }

    __aicore__ inline void DataCopyIn(SequenceParams* params, int cnt)
    {
        LocalTensor<int32_t> ts = queTimestamps.AllocTensor<int32_t>();
        for (int i = 0; i < cnt; ++i) {
            SequenceParams param = params[i];
            int startIndexGT = param.startIndexGT;
            int startIndexUb = param.startIndexUb;

            DataCopy(ts[startIndexUb], timestampsGT[startIndexGT], alignSeqLen);
        }
        queTimestamps.EnQue(ts);
    }

    __aicore__ inline void ComputeBucketTimestamps(SequenceParams* params, int rowCnt)
    {
        LocalTensor<int32_t> tsInt = queTimestamps.DeQue<int32_t>();
        LocalTensor<float> tsTmp = tsInt.template ReinterpretCast<float>();
        LocalTensor<float> ts = queTimestampsFloat.AllocTensor<float>();
        LocalTensor<uint8_t> buff = tmpQue.AllocTensor<uint8_t>();

        for (int i = 0; i < rowCnt; ++i) {
            SequenceParams param = params[i];
            int startIndexUb = param.startIndexUb;
            int value = param.subValue;
            Adds(tsInt[startIndexUb], tsInt[startIndexUb], (int32_t)-value, s);
        }

        uint32_t cnt = rowCnt * alignSeqLen;
        Cast(ts, tsInt, RoundMode::CAST_NONE, cnt);

        Abs(ts, ts, cnt);
        ClampMin(tsTmp, ts, buff, clampMin, cnt);
        Log(ts, tsTmp, cnt);
        Muls(ts, ts, div, cnt);
        ClampMax(tsTmp, ts, buff, (float)numBuckets, cnt);

        Cast(tsInt, tsTmp, RoundMode::CAST_TRUNC, cnt);

        tmpQue.FreeTensor(buff);
        queTimestampsFloat.FreeTensor(ts);
        queTimestamps.EnQue(tsInt);
    }

    __aicore__ inline void IndexSelect(LocalTensor<FloatType>& tsw, LocalTensor<uint32_t>& tsInt, int layer, int rowCnt)
    {
        uint32_t cnt = rowCnt * alignSeqLen;
        LocalTensor<FloatType> rabTime = queTimestampsFloat.AllocTensor<FloatType>();
        uint32_t processLenMax = GATHER_PROCESS_WINDOW / sizeof(FloatType);
        uint32_t tmpOffset = 0;
        while (tmpOffset < cnt) {
            uint32_t processLen = (cnt - tmpOffset) > processLenMax ? processLenMax : (cnt - tmpOffset);
            Gather(rabTime[tmpOffset], tsw[layer * alignNumBuckets], tsInt[tmpOffset], (uint32_t)0, processLen);
            tmpOffset += processLen;
        }
        queTimestampsFloat.EnQue(rabTime);
    }

    __aicore__ inline void DataCopyOut(uint32_t ptr, int rowCnt)
    {
        LocalTensor<FloatType> rabTime = queTimestampsFloat.DeQue<FloatType>();

        for (int i = 0; i < rowCnt; ++i) {
            uint32_t ptrUb = i * alignSeqLen;

            // 对齐部分拷出
            if (alignLen > 0) {
                DataCopy(rabTimeBiasOutGT[ptr + i * s], rabTime[ptrUb], s);
            }
            // 非对齐拷出
            if (unalignLen > 0) {
#ifdef SUPPORT_V200
                uint64_t mask0 = (1ul << (DATA_ALIGN_BYTES / sizeof(FloatType))) - (1ul << unalignCnt);
                uint64_t mask[2] = {mask0, 0};
                Duplicate(rabTime[ptrUb + alignCnt], (FloatType)0, mask, 1, 1, 1);
                queTimestampsFloat.EnQue(rabTime);
                rabTime = queTimestampsFloat.DeQue<FloatType>();
                SetAtomicAdd<FloatType>();
                DataCopy(rabTimeBiasOutGT[ptr + i * s + alignCnt], rabTime[ptrUb + alignCnt],
                         Ceil(unalignLen) / sizeof(FloatType));
                SetAtomicNone();
#else
                const DataCopyExtParams dataCopyExtParams{1, unalignLen, 0, 0, 0};
                DataCopyPad(rabTimeBiasOutGT[ptr + i * s + alignCnt], rabTime[ptrUb + alignCnt], dataCopyExtParams);
#endif
            }
        }
        queTimestampsFloat.FreeTensor(rabTime);
    }

    __aicore__ inline void DataCopyOutIndex(LocalTensor<int32_t>& bucketTimestamps, uint32_t rowOffset, uint32_t rowCnt)
    {
        uint32_t seqDatasize = s * sizeof(int32_t);
        uint32_t alignLen32 = seqDatasize / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t alignCnt32 = alignLen32 / sizeof(int32_t);
        uint32_t unalignLen32 = seqDatasize - alignLen32;

        uint32_t ptr = rowOffset * s;
        uint32_t ptrUb = 0;

        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rowCnt; ++i) {
            // 对齐部分拷出
            if (alignLen32 > 0) {
                DataCopy(bucketTimestampsOutGT[ptr], bucketTimestamps[ptrUb], alignCnt32);
            }
            // 非对齐拷出
            if (unalignLen32 > 0) {
                const DataCopyExtParams dataCopyExtParams{1, unalignLen32, 0, 0, 0};
                DataCopyPad(bucketTimestampsOutGT[ptr + alignCnt32],
                            bucketTimestamps[ptrUb + alignCnt32], dataCopyExtParams);
            }
            ptr += s;
            ptrUb += alignSeqLen;
        }

        TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void DataCopyInTsw()
    {
        LocalTensor<FloatType> tsw = queTimestampsWeights.AllocTensor<FloatType>();
        for (int n = 0; n < numLayer; ++n) {
            DataCopy(tsw[n * alignNumBuckets], timestampsWeightsGT[n * numBuckets], alignNumBuckets);
        }
        queTimestampsWeights.EnQue(tsw);
    }

    __aicore__ inline int64_t Ceil(int64_t a, int64_t b = DATA_ALIGN_BYTES)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b * b;
    }

    __aicore__ inline void Compute(Args args)
    {
        Init(args);
        DataCopyInTsw();
        LocalTensor<FloatType> tsw = queTimestampsWeights.DeQue<FloatType>();
        SequenceParams params[MAX_SEQ_CNT];
        for (int offset = 0; offset < processRowLen; offset += stride) {
            int rowOffset = offset + startIndex;
            int rowCnt = stride > (processRowLen - offset) ? (processRowLen - offset) : stride;

            FillSeqParams(params, rowOffset, rowCnt);
            DataCopyIn(params, rowCnt);
            ComputeBucketTimestamps(params, rowCnt);

            LocalTensor<int32_t> bucketTimestamps = queTimestamps.DeQue<int32_t>();
#ifndef SUPPORT_V200
            DataCopyOutIndex(bucketTimestamps, rowOffset, rowCnt);
#endif
            // 计算gather时的偏移量单位为bytes
            Muls(bucketTimestamps, bucketTimestamps, (int32_t)sizeof(FloatType), rowCnt * alignSeqLen);
            queTimestamps.EnQue(bucketTimestamps);
            bucketTimestamps = queTimestamps.DeQue<int32_t>();
            LocalTensor<uint32_t> index = bucketTimestamps.template ReinterpretCast<uint32_t>();

            for (int n = 0; n < numLayer; ++n) {
                IndexSelect(tsw, index, n, rowCnt);
                pipe_barrier(PIPE_ALL);

                uint32_t ptr = (n * bs * s + rowOffset) * s;
                DataCopyOut(ptr, rowCnt);
            }
            queTimestamps.FreeTensor(index);
        }
        queTimestampsWeights.FreeTensor(tsw);
    }

private:
    // shape
    uint32_t s;
    uint32_t alignSeqLen;
    uint32_t bs;
    uint32_t stride;
    // align
    uint32_t alignLen;
    uint32_t alignCnt;
    uint32_t unalignLen;
    uint32_t unalignCnt;
    // tiling
    uint32_t startIndex;
    uint32_t processRowLen;

    float div;
    int32_t numBuckets;
    int32_t alignNumBuckets;
    int32_t numLayer;
    float clampMin;
    float clampMax;

private:
    GlobalTensor<int32_t> timestampsGT;
    GlobalTensor<FloatType> timestampsWeightsGT;
    GlobalTensor<FloatType> rabTimeBiasOutGT;
    GlobalTensor<int32_t> bucketTimestampsOutGT;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> queTimestamps;
    TQue<TPosition::VECOUT, 1> queTimestampsFloat;
    TQue<TPosition::VECIN, 1> queTimestampsWeights;
    TQue<TPosition::VECCALC, 1> tmpQue;
};
#endif  // MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TIME_H
