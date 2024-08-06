/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef NORMALIZE_COMPUTE__H
#define NORMALIZE_COMPUTE__H
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

namespace Attention_Kernel {

constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALREADY_ALIGNED = 1;
constexpr int32_t SPECIAL_CASE = 2;
constexpr int32_t PAD_SIZE = (16 * 56 * 16);
constexpr int32_t SPECIAL_BLOCK_COUNT = 16;
constexpr int32_t SPECIAL_BLOCK_LEN = (50 * 16 / 8);
constexpr int32_t SPECIAL_STRIDE = (6 * 16 / 8);
constexpr int32_t PAD_VALUE = -1000;
constexpr float MASK_MUL_CONST = -10000;
constexpr float MASK_ADD_CONST = 10000;

struct NormalizeArgs {
    TPipe* pipe;

    uint8_t attr;
    int queryDim1;
    int keyDim1;
    int loopCount;
    int normalizeRow;
    int normalizeColumn;
    int maskIsOn;
    float normalizeSqrt;
    uint64_t maxSharedTmpBuf;

    const SoftMaxTiling* tiling;

    const ConfusionTransposeTiling* tilingData;
    const ConfusionTransposeTiling* tilingData1;
    const ConfusionTransposeTiling* tilingData2;
    const ConfusionTransposeTiling* tilingData3;
};

template<typename qType>
class NormalizeCompute {
public:
    __aicore__ inline NormalizeCompute() {}

    __aicore__ inline void Init(NormalizeArgs normalArgs)
    {
        this->args = normalArgs;
        int bufSize = args.normalizeRow * args.normalizeColumn * sizeof(qType);
        args.pipe->InitBuffer(vecInQueue, 1, bufSize);
        args.pipe->InitBuffer(vecOutQueue, 1, bufSize);
        args.pipe->InitBuffer(vecSharedQueue, 1, args.maxSharedTmpBuf);
    }

    __aicore__ inline void DoPadLocal(LocalTensor<qType>& sourceTensor, LocalTensor<qType>& mindTensor,
                                        const ConfusionTransposeTiling* tilingData,
                                        const ConfusionTransposeTiling* tilingData1)
    {
        ConfusionTransposeTiling tiling = *tilingData;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, PAD_VALUE, PAD_SIZE);
        DataCopyParams dataCopyParam = {SPECIAL_BLOCK_COUNT, SPECIAL_BLOCK_LEN, 0, SPECIAL_STRIDE};
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *tilingData1;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, PAD_SIZE);
    }

    __aicore__ inline void DoUnPadLocal(LocalTensor<qType>& sourceTensor, LocalTensor<qType>& mindTensor,
                                        const ConfusionTransposeTiling* tilingData2,
                                        const ConfusionTransposeTiling* tilingData3)
    {
        ConfusionTransposeTiling tiling = *tilingData2;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, PAD_VALUE, PAD_SIZE);
        DataCopyParams dataCopyParam = {SPECIAL_BLOCK_COUNT, SPECIAL_BLOCK_LEN, SPECIAL_STRIDE, 0};
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *tilingData3;
        ConfusionTranspose<qType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, PAD_SIZE);
    }

    __aicore__ inline void Process(GlobalTensor<qType> softmaxGlobleTensor, GlobalTensor<qType> softmaxGbMask)
    {
        srcGloblePtr = softmaxGlobleTensor;
        maskGloblePtr = softmaxGbMask;
        offset = 0;
        usedRowCount = 0;
        uint8_t padLen = args.normalizeColumn - args.keyDim1;
        padParams = {false, 0, padLen, 0};

        for (int i = 0; i < args.loopCount; i++) {
            /* Get height of softmax matrix and handle the last loop height */
            height = ((args.queryDim1 - usedRowCount) < args.normalizeRow) ?
                        args.queryDim1 - usedRowCount : args.normalizeRow;
            totalSize = height * args.normalizeColumn;

            CopyIn();
            PreCompute();
            Compute();
            CopyOut();

            usedRowCount += height;
            offset += args.normalizeRow * args.keyDim1;
        }
    }

    __aicore__ inline void CopyMask()
    {
        LocalTensor<qType> LocalMask = vecSharedQueue.AllocTensor<qType>();
        if (args.attr == ALREADY_ALIGNED) {
            DataCopy(LocalMask, maskGloblePtr[offset], totalSize);
        } else if (args.attr == SPECIAL_CASE) {
            DataCopy(LocalMask, maskGloblePtr[offset], totalSize);
        } else {
            DataCopyPad(LocalMask, maskGloblePtr[offset], copyParams, padParams);
        }

        vecSharedQueue.EnQue(LocalMask);
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<qType> inLocalTensor = vecInQueue.AllocTensor<qType>();
        LocalTensor<qType> outLocalTensor = vecOutQueue.AllocTensor<qType>();

        if (args.attr == ALREADY_ALIGNED) {
            DataCopy(inLocalTensor, srcGloblePtr[offset], totalSize);
        } else if (args.attr == SPECIAL_CASE) {
            DataCopy(inLocalTensor, srcGloblePtr[offset], totalSize);
        } else {
            copyParams.blockCount = height;
            copyParams.blockLen = args.keyDim1 * sizeof(qType);
            DataCopyPad(inLocalTensor, srcGloblePtr[offset], copyParams, padParams);
        }

        if (args.maskIsOn == 1) {
            CopyMask();
        }

        vecInQueue.EnQue(inLocalTensor);
        vecOutQueue.EnQue(outLocalTensor);
    }

    __aicore__ inline void PreCompute()
    {
        LocalTensor<qType> inLocalTensor = vecInQueue.DeQue<qType>();
        LocalTensor<qType> outLocalTensor = vecOutQueue.DeQue<qType>();

        if (args.attr == SPECIAL_CASE && args.maskIsOn == 1) {
            LocalTensor<qType> LocalMask = vecSharedQueue.DeQue<qType>();
            DoPadLocal(LocalMask, outLocalTensor, args.tilingData, args.tilingData1);
            DoPadLocal(inLocalTensor, outLocalTensor, args.tilingData, args.tilingData1);
            vecSharedQueue.EnQue(LocalMask);
        } else if (args.attr == SPECIAL_CASE) {
            DoPadLocal(inLocalTensor, outLocalTensor, args.tilingData, args.tilingData1);
        }

        // atten_weight = qkMatMul / sqrt(atten_dim)
        Muls(inLocalTensor, inLocalTensor, args.normalizeSqrt, totalSize);

        if (args.maskIsOn == 1) {
            LocalTensor<qType> LocalMask = vecSharedQueue.DeQue<qType>();
            // atten_mask = (1 - mask) * 10000
            Muls(LocalMask, LocalMask, MASK_MUL_CONST, totalSize);
            Adds(LocalMask, LocalMask, MASK_ADD_CONST, totalSize);

            // atten_weight = atten_weight + atten_mask
            Add(inLocalTensor, inLocalTensor, LocalMask, totalSize);
            vecSharedQueue.FreeTensor(LocalMask);
        }

        vecInQueue.EnQue<qType>(inLocalTensor);
        vecOutQueue.EnQue<qType>(outLocalTensor);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<qType> inLocalTensor = vecInQueue.DeQue<qType>();
        LocalTensor<qType> outLocalTensor = vecOutQueue.DeQue<qType>();
        LocalTensor<uint8_t> sharedTmpBuf = vecSharedQueue.AllocTensor<uint8_t>();

        uint32_t weightPad = args.normalizeColumn;
        uint32_t weight = args.keyDim1;
        SoftMaxShapeInfo scrShape = {height, weightPad, height, weight};
        SoftMax<qType>(outLocalTensor, inLocalTensor, sharedTmpBuf, *args.tiling, scrShape);

        if (args.attr == SPECIAL_CASE) {
            DoUnPadLocal(outLocalTensor, inLocalTensor, args.tilingData2, args.tilingData3);
        }
        vecOutQueue.EnQue<qType>(outLocalTensor);
        vecInQueue.FreeTensor(inLocalTensor);
        vecSharedQueue.FreeTensor(sharedTmpBuf);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<qType> outLocalTensor = vecOutQueue.DeQue<qType>();

        if (args.attr == ALREADY_ALIGNED) {
            DataCopy(srcGloblePtr[offset], outLocalTensor, totalSize);
        } else if (args.attr == SPECIAL_CASE) {
            uint32_t thisLen = height * args.keyDim1 * sizeof(qType);
            if ((thisLen % ALIGN_32) != 0) {
                DataCopyExtParams dataCopyParamTail {1, thisLen, 0, 0, 0};
                DataCopyPad(srcGloblePtr[offset], outLocalTensor, dataCopyParamTail);
            } else {
                DataCopy(srcGloblePtr[offset], outLocalTensor, height * args.keyDim1);
            }
        } else {
            DataCopyPad(srcGloblePtr[offset], outLocalTensor, copyParams);
        }
        vecOutQueue.FreeTensor(outLocalTensor);
    }

private:
    NormalizeArgs args;
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> vecSharedQueue;

    GlobalTensor<qType> srcGloblePtr;
    GlobalTensor<qType> maskGloblePtr;
    uint32_t height = 0;
    int offset = 0;
    int usedRowCount = 0;
    uint32_t totalSize = 0;
    struct DataCopyExtParams copyParams;
    struct DataCopyPadExtParams<qType> padParams;
};
}
#endif