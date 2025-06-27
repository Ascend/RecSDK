/**
 * @file relative_attn_bias_backward.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_RELATIVE_ATTN_BIAS_BACKWARD_H
#define MXREC_RELATIVE_ATTN_BIAS_BACKWARD_H
#include <type_traits>
#include "rab_common.h"
#include "kernel_operator.h"

template <typename FloatType>
class RelativeAttnBiasBackward {
public:
    __aicore__ inline RelativeAttnBiasBackward() {}

    __aicore__ inline void InitTensor(Args args)
    {
        tsGradGT.SetGlobalBuffer((__gm__ FloatType*)args.rabTimeGrad, numLayer * bs * s * s);
        bucketTimestampsGT.SetGlobalBuffer((__gm__ int32_t*)args.bucketTimestamps, bs * s * s);
        tswGradOutGT.SetGlobalBuffer((__gm__ FloatType*)args.timestampsWeightsGrad, tswTableSize);

        pipe.InitBuffer(inQueTsGrad, 1, AlignTo32(stride * sizeof(float)));
        pipe.InitBuffer(inQueBucketTimestamps, 1, AlignTo32(stride * sizeof(int32_t)));
        pipe.InitBuffer(outQueTswGradOut, 1, AlignTo32(numBuckets * numLayer * sizeof(float)));
        if (std::is_same<FloatType, half>::value) {
            pipe.InitBuffer(tmpQue, 1, AlignTo32(stride * sizeof(FloatType)));
        }
    }

    __aicore__ inline void InitTiling()
    {
        int totalLen = bs * s * s;
        int totalTableSizeSplit = totalLen % GetBlockNum();
        int baseLen = totalLen / GetBlockNum();
        // 计算总共要处理的数据量、数据起始位置
        if (GetBlockIdx() >= totalTableSizeSplit) {
            processLen = baseLen;
            startGT = totalTableSizeSplit * (baseLen + 1) + (GetBlockIdx() - totalTableSizeSplit) * baseLen;
        } else {
            processLen = baseLen + 1;
            startGT = GetBlockIdx() * (baseLen + 1);
        }
    }

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        s = tilingData.s;
        bs = tilingData.bs;
        stride = tilingData.timeStride;
        numBuckets = tilingData.numBuckets;
        numLayer = tilingData.numLayer;
        tswTableSize = numLayer * numBuckets;

        InitTensor(args);
        InitTiling();
    }

    __aicore__ inline void InitTswGrad()
    {
        LocalTensor<float> gradOut = outQueTswGradOut.AllocTensor<float>();
        Duplicate(gradOut, (float)0, AlignTo32(numLayer * numBuckets * sizeof(float)) / sizeof(float));
        outQueTswGradOut.EnQue(gradOut);
    }

    __aicore__ inline void DataCopyInIndex(uint32_t offset, uint32_t cnt)
    {
        LocalTensor<int32_t> bucketTimestamps = inQueBucketTimestamps.AllocTensor<int32_t>();
        DataCopy(bucketTimestamps, bucketTimestampsGT[offset], AlignTo32(cnt * sizeof(int32_t)) / sizeof(int32_t));
        inQueBucketTimestamps.EnQue(bucketTimestamps);
    }

    __aicore__ inline void DataCopyInGrad(uint32_t layer, uint32_t offset, uint32_t cnt)
    {
        if (std::is_same<FloatType, half>::value) {
            // 数据拷入
            LocalTensor<FloatType> gradFP16 = tmpQue.AllocTensor<FloatType>();
            DataCopy(gradFP16, tsGradGT[offset + layer * bs * s * s],
                     AlignTo32(cnt * sizeof(FloatType)) / sizeof(FloatType));
            tmpQue.EnQue(gradFP16);
            gradFP16 = tmpQue.DeQue<FloatType>();
            // 数据转换
            LocalTensor<float> gradFP32 = inQueTsGrad.AllocTensor<float>();
            Cast(gradFP32, gradFP16, RoundMode::CAST_NONE, cnt);

            inQueTsGrad.EnQue(gradFP32);
            tmpQue.FreeTensor(gradFP16);
        } else {
            // 数据拷入
            LocalTensor<FloatType> gradFP32 = inQueTsGrad.AllocTensor<FloatType>();
            DataCopy(gradFP32, tsGradGT[offset + layer * bs * s * s],
                     AlignTo32(cnt * sizeof(FloatType)) / sizeof(FloatType));
            inQueTsGrad.EnQue(gradFP32);
        }
    }

    __aicore__ inline void ScatterAdd(LocalTensor<float>& dst,
                                      LocalTensor<float>& src,
                                      LocalTensor<int32_t>& index,
                                      uint32_t layer,
                                      uint32_t cnt)
    {
        uint32_t layerOffset = layer * numBuckets;
        __ubuf__ float* dstAddr = reinterpret_cast<__ubuf__ float*>(dst[layerOffset].GetPhyAddr());
        __ubuf__ float* srcAddr = reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr());
        __ubuf__ int32_t* indexAddr = reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr());
        for (int i = 0; i < cnt; ++i) {
            const auto ind = indexAddr[i];
            const auto value = srcAddr[i];
            dstAddr[ind] += value;
        }
    }

    __aicore__ inline void DataCopyOut(LocalTensor<float>& gradOut)
    {
        // 同步计算结果
        uint32_t alignTswTableSize = AlignTo32(tswTableSize * sizeof(FloatType)) / sizeof(FloatType);
        outQueTswGradOut.EnQue(gradOut);

        if (std::is_same<FloatType, half>::value) {
            LocalTensor<float> gradOutFP32 = outQueTswGradOut.DeQue<float>();
            LocalTensor<FloatType> gradOutFP16 = tmpQue.AllocTensor<FloatType>();
            Cast(gradOutFP16, gradOutFP32, RoundMode::CAST_ROUND, tswTableSize);
            tmpQue.EnQue(gradOutFP16);
            gradOutFP16 = tmpQue.DeQue<FloatType>();

            SetAtomicAdd<FloatType>();
            DataCopy(tswGradOutGT, gradOutFP16, alignTswTableSize);
            SetAtomicNone();

            tmpQue.FreeTensor(gradOutFP16);
        } else if (std::is_same<FloatType, float>::value) {
            LocalTensor<FloatType> gradOutFP32 = outQueTswGradOut.DeQue<FloatType>();
            SetAtomicAdd<FloatType>();
            DataCopy(tswGradOutGT, gradOutFP32, alignTswTableSize);
            SetAtomicNone();
        }
    }

    __aicore__ inline void Compute(Args args)
    {
        Init(args);
        InitTswGrad();

        uint32_t offset = 0;
        LocalTensor<float> gradOut = outQueTswGradOut.DeQue<float>();
        while (offset < processLen) {
            uint32_t remain = processLen - offset;
            uint32_t cnt = remain > stride ? stride : remain;

            DataCopyInIndex(startGT + offset, cnt);
            LocalTensor<int32_t> index = inQueBucketTimestamps.DeQue<int32_t>();
            for (uint32_t n = 0; n < numLayer; ++n) {
                DataCopyInGrad(n, startGT + offset, cnt);
                pipe_barrier(PIPE_ALL);

                LocalTensor<float> grad = inQueTsGrad.DeQue<float>();
                ScatterAdd(gradOut, grad, index, n, cnt);
                pipe_barrier(PIPE_ALL);

                inQueTsGrad.FreeTensor(grad);
            }
            inQueBucketTimestamps.FreeTensor(index);
            offset += cnt;
        }
        DataCopyOut(gradOut);
        outQueTswGradOut.FreeTensor(gradOut);
    }

private:
    GlobalTensor<FloatType> tsGradGT;
    GlobalTensor<int32_t> bucketTimestampsGT;
    GlobalTensor<FloatType> tswGradOutGT;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> tmpQue;
    TQue<TPosition::VECIN, 1> inQueTsGrad;
    TQue<TPosition::VECIN, 1> inQueBucketTimestamps;
    TQue<TPosition::VECOUT, 1> outQueTswGradOut;

private:
    // shape
    uint32_t s;
    uint32_t bs;
    uint32_t stride;
    uint32_t numBuckets;
    uint32_t numLayer;
    uint32_t tswTableSize;
    // tiling
    uint32_t processLen;
    uint32_t startGT;
};
#endif  // MXREC_RELATIVE_ATTN_BIAS_BACKWARD_H
