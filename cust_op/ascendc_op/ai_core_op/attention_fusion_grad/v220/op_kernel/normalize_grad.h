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

#ifndef NORMALIZE_GRAD_COMPUTE_H
#define NORMALIZE_GRAD_COMPUTE_H

#include "args.h"
#include "kernel_operator.h"
#include "utils.h"
using namespace AscendC;

namespace AscendFusionGrad {

template <typename tType>
class NormalizeGrad {
public:
    __aicore__ inline NormalizeGrad() {}

    __aicore__ inline int GetAlignSize()
    {
        int ubAlign = 32;
        return ubAlign / sizeof(tType);
    }

    __aicore__ inline void Init(AttentionFusionGradArgs args, NormGradPipeArgs pipeArgs)
    {
        this->args = args;

        softmaxShpeOfOneBatch = args.shapeArgs.queryDim1 * args.shapeArgs.keyDim1;

        softmaxOut.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.softmaxOut),
                                   args.shapeArgs.batchNum * softmaxShpeOfOneBatch);

        gradSoftmax.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.inputArgs.workspace),
                                    args.shapeArgs.batchNum * softmaxShpeOfOneBatch);

        vecInQueue = pipeArgs.vecInQueue;
        vecInGradQueue = pipeArgs.vecInGradQueue;
        vecOutQueue = pipeArgs.vecOutQueue;
        tmpBuff = pipeArgs.tmpBuff;
    }

    __aicore__ inline void DoPadLocal(LocalTensor<tType>& sourceTensor, LocalTensor<tType>& mindTensor,
                                      const ConfusionTransposeTiling* confusionTransposeTilingData,
                                      const ConfusionTransposeTiling* confusionTransposeTilingData1)
    {
        int keyDim1 = args.shapeArgs.keyDim1;
        int paddingKeyDim1 = args.shapeTilingArgs.paddingKeyDim1;
        int transposeAlignDim = args.shapeTilingArgs.transposeAlignDim;
        int paddingLen = paddingKeyDim1 - args.shapeArgs.keyDim1;

        int totalSize = TRANSPOSE_ALIGNMENT * keyDim1 * transposeAlignDim;
        int padSize = TRANSPOSE_ALIGNMENT * paddingKeyDim1 * transposeAlignDim;

        ConfusionTransposeTiling tiling = *confusionTransposeTilingData;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, 0, padSize);
        DataCopyParams dataCopyParam = {0, 0, 0, 0};
        dataCopyParam.blockCount = transposeAlignDim;
        dataCopyParam.blockLen = keyDim1 * TRANSPOSE_ALIGNMENT / GetAlignSize();
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = paddingLen * TRANSPOSE_ALIGNMENT / GetAlignSize();

        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *confusionTransposeTilingData1;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, padSize);
    }

    __aicore__ inline void DoUnPadLocal(LocalTensor<tType>& sourceTensor, LocalTensor<tType>& mindTensor,
                                        const ConfusionTransposeTiling* confusionTransposeTilingData2,
                                        const ConfusionTransposeTiling* confusionTransposeTilingData3)
    {
        int keyDim1 = args.shapeArgs.keyDim1;
        int paddingKeyDim1 = args.shapeTilingArgs.paddingKeyDim1;
        int transposeAlignDim = args.shapeTilingArgs.transposeAlignDim;
        int paddingLen = paddingKeyDim1 - args.shapeArgs.keyDim1;

        int totalSize = TRANSPOSE_ALIGNMENT * keyDim1 * transposeAlignDim;
        int padSize = TRANSPOSE_ALIGNMENT * paddingKeyDim1 * transposeAlignDim;

        ConfusionTransposeTiling tiling = *confusionTransposeTilingData2;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
        Duplicate<float>(sourceTensor, 0, padSize);
        DataCopyParams dataCopyParam = {0, 0, 0, 0};
        dataCopyParam.blockCount = transposeAlignDim;
        dataCopyParam.blockLen = keyDim1 * TRANSPOSE_ALIGNMENT / GetAlignSize();
        dataCopyParam.srcStride = paddingLen * TRANSPOSE_ALIGNMENT / GetAlignSize();
        dataCopyParam.dstStride = 0;
        DataCopy(sourceTensor, mindTensor, dataCopyParam);

        ConfusionTransposeTiling tiling1 = *confusionTransposeTilingData3;
        ConfusionTranspose<tType>(mindTensor, sourceTensor, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling1);
        DataCopy(sourceTensor, mindTensor, padSize);
    }

    __aicore__ inline void ProcessOneBatch(uint32_t batchI)
    {
        batchI += args.shapeTilingArgs.batchOffset;
        if (args.shapeTilingArgs.keyDim1Align == KEY_DIM1_COPY_ALIGN_MODE) {
            OneBatchWithDataCopyAlign(batchI);
        } else if (args.shapeTilingArgs.keyDim1Align == KEY_DIM1_COPY_TRANSPOSE_ALIGN_MODE) {
            OneBatchWithTransposeAlign(batchI);
        } else if (args.shapeTilingArgs.keyDim1Align == KEY_DIM1_COPY_PAD_MODE) {
            OneBatchWithDataCopyPad(batchI);
        }
    }

    __aicore__ inline void OneBatchWithTransposeAlign(uint32_t batchI)
    {
        GlobalTensor<tType> thisBatchSoftmaxGb = softmaxOut[batchI * softmaxShpeOfOneBatch];
        GlobalTensor<tType> thisBatchSoftmaxGradGb = gradSoftmax[batchI * softmaxShpeOfOneBatch];

        int total = softmaxShpeOfOneBatch;
        int remain = total;

        while (remain > 0) {
            // Caculate basic
            int thisLen = args.shapeTilingArgs.numRowOfNormalizeOne * args.shapeArgs.keyDim1;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int offset = total - remain;

            LocalTensor<tType> inLocalTensor = vecInQueue->AllocTensor<tType>();
            LocalTensor<tType> inGradLocalTensor = vecInGradQueue->AllocTensor<tType>();

            DataCopy(inGradLocalTensor, thisBatchSoftmaxGradGb[offset], thisLen);
            DataCopy(inLocalTensor, thisBatchSoftmaxGb[offset], thisLen);

            vecInQueue->EnQue(inLocalTensor);
            vecInGradQueue->EnQue(inGradLocalTensor);

            LocalTensor<tType> inLocalTensorCompute = vecInQueue->DeQue<tType>();
            LocalTensor<tType> inGradLocalTensorCompute = vecInGradQueue->DeQue<tType>();
            LocalTensor<tType> outLocalTensor = vecOutQueue->AllocTensor<tType>();

            DoPadLocal(inLocalTensorCompute, outLocalTensor, args.tilingArgs.unAlign2AlignStep1Tiling,
                       args.tilingArgs.unAlign2AlignStep2Tiling);
            DoPadLocal(inGradLocalTensorCompute, outLocalTensor, args.tilingArgs.unAlign2AlignStep1Tiling,
                       args.tilingArgs.unAlign2AlignStep2Tiling);

            uint32_t height = thisLen / args.shapeArgs.keyDim1;
            uint32_t weightPadding = args.shapeTilingArgs.paddingKeyDim1;
            uint32_t weightOrig = args.shapeArgs.keyDim1;

            SoftMaxShapeInfo scrShape = {height, weightPadding, height, weightOrig};

            SoftmaxGrad<tType>(outLocalTensor, inGradLocalTensorCompute, inLocalTensorCompute, tmpBuff->Get<uint8_t>(),
                               *args.tilingArgs.softmaxtiling, false, scrShape);
            Muls(outLocalTensor, outLocalTensor, args.inputArgs.attenDimSqrt, height * weightPadding);

            DoUnPadLocal(outLocalTensor, inLocalTensorCompute, args.tilingArgs.Align2UnAlignStep1Tiling,
                         args.tilingArgs.Align2UnAlignStep2Tiling);
            vecInQueue->FreeTensor(inLocalTensorCompute);
            vecInGradQueue->FreeTensor(inGradLocalTensorCompute);

            vecOutQueue->EnQue<tType>(outLocalTensor);
            LocalTensor<tType> copyOutLocalTensor = vecOutQueue->DeQue<tType>();

            uint32_t copyLenAlign = thisLen / GetAlignSize() * GetAlignSize();
            uint32_t remainOfThisCopy = thisLen - copyLenAlign;
            DataCopy(thisBatchSoftmaxGradGb[offset], copyOutLocalTensor, copyLenAlign);
            if (remainOfThisCopy != 0) {
                DataCopyExtParams dataCopyParamTail{1, remainOfThisCopy * static_cast<uint32_t>(sizeof(tType)), 0, 0,
                                                    0};
                DataCopyPad(thisBatchSoftmaxGradGb[offset + copyLenAlign], copyOutLocalTensor, dataCopyParamTail);
            }

            vecOutQueue->FreeTensor(copyOutLocalTensor);
            remain = remain - thisLen;
        }
    }

    __aicore__ inline void OneBatchWithDataCopyAlign(uint32_t batchI)
    {
        GlobalTensor<tType> thisBatchSoftmaxGb = softmaxOut[batchI * softmaxShpeOfOneBatch];
        GlobalTensor<tType> thisBatchSoftmaxGradGb = gradSoftmax[batchI * softmaxShpeOfOneBatch];

        int total = softmaxShpeOfOneBatch;
        int remain = total;

        while (remain > 0) {
            // Caculate basic
            int thisLen = args.shapeTilingArgs.numRowOfNormalizeOne * args.shapeArgs.keyDim1;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int offset = total - remain;

            LocalTensor<tType> inLocalTensor = vecInQueue->AllocTensor<tType>();
            LocalTensor<tType> inGradLocalTensor = vecInGradQueue->AllocTensor<tType>();

            DataCopy(inGradLocalTensor, thisBatchSoftmaxGradGb[offset], thisLen);
            DataCopy(inLocalTensor, thisBatchSoftmaxGb[offset], thisLen);

            vecInQueue->EnQue(inLocalTensor);
            vecInGradQueue->EnQue(inGradLocalTensor);

            LocalTensor<tType> inLocalTensorCompute = vecInQueue->DeQue<tType>();
            LocalTensor<tType> inGradLocalTensorCompute = vecInGradQueue->DeQue<tType>();
            LocalTensor<tType> outLocalTensor = vecOutQueue->AllocTensor<tType>();

            uint32_t height = thisLen / args.shapeArgs.keyDim1;
            uint32_t weightPadding = args.shapeTilingArgs.paddingKeyDim1;
            uint32_t weightOrig = args.shapeArgs.keyDim1;

            SoftMaxShapeInfo scrShape = {height, weightPadding, height, weightOrig};

            SoftmaxGrad<tType>(outLocalTensor, inGradLocalTensorCompute, inLocalTensorCompute, tmpBuff->Get<uint8_t>(),
                               *args.tilingArgs.softmaxtiling, false, scrShape);
            Muls(outLocalTensor, outLocalTensor, args.inputArgs.attenDimSqrt, height * weightPadding);

            vecInQueue->FreeTensor(inLocalTensorCompute);
            vecInGradQueue->FreeTensor(inGradLocalTensorCompute);

            vecOutQueue->EnQue<tType>(outLocalTensor);
            LocalTensor<tType> copyOutLocalTensor = vecOutQueue->DeQue<tType>();

            DataCopy(thisBatchSoftmaxGradGb[offset], copyOutLocalTensor, thisLen);

            vecOutQueue->FreeTensor(copyOutLocalTensor);
            remain = remain - thisLen;
        }
    }

    __aicore__ inline void OneBatchWithDataCopyPad(uint32_t batchI)
    {
        struct DataCopyExtParams copyParams {
            0, 0, 0, 0, 0
        };
        DataCopyPadExtParams<tType> padParams{
            true, 0, (uint8_t)(args.shapeTilingArgs.paddingKeyDim1 - args.shapeArgs.keyDim1), 0};
        GlobalTensor<tType> thisBatchSoftmaxGb = softmaxOut[batchI * softmaxShpeOfOneBatch];
        GlobalTensor<tType> thisBatchSoftmaxGradGb = gradSoftmax[batchI * softmaxShpeOfOneBatch];

        int total = softmaxShpeOfOneBatch;
        int remain = total;

        while (remain > 0) {
            int thisLen = args.shapeTilingArgs.numRowOfNormalizeOne * args.shapeArgs.keyDim1;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int offset = total - remain;

            copyParams.blockCount = thisLen / args.shapeArgs.keyDim1;
            copyParams.blockLen = args.shapeArgs.keyDim1 * sizeof(tType);

            LocalTensor<tType> inLocalTensor = vecInQueue->AllocTensor<tType>();
            LocalTensor<tType> inGradLocalTensor = vecInGradQueue->AllocTensor<tType>();

            DataCopyPad(inGradLocalTensor, thisBatchSoftmaxGradGb[offset], copyParams, padParams);
            DataCopyPad(inLocalTensor, thisBatchSoftmaxGb[offset], copyParams, padParams);

            vecInQueue->EnQue(inLocalTensor);
            vecInGradQueue->EnQue(inGradLocalTensor);

            LocalTensor<tType> inLocalTensorCompute = vecInQueue->DeQue<tType>();
            LocalTensor<tType> inGradLocalTensorCompute = vecInGradQueue->DeQue<tType>();
            LocalTensor<tType> outLocalTensor = vecOutQueue->AllocTensor<tType>();

            uint32_t height = thisLen / args.shapeArgs.keyDim1;
            uint32_t weightPadding = args.shapeTilingArgs.paddingKeyDim1;
            uint32_t weightOrig = args.shapeArgs.keyDim1;

            SoftMaxShapeInfo scrShape = {height, weightPadding, height, weightOrig};

            SoftmaxGrad<tType>(outLocalTensor, inGradLocalTensorCompute, inLocalTensorCompute, tmpBuff->Get<uint8_t>(),
                               *args.tilingArgs.softmaxtiling, false, scrShape);
            Muls(outLocalTensor, outLocalTensor, args.inputArgs.attenDimSqrt, height * weightPadding);

            vecInQueue->FreeTensor(inLocalTensorCompute);
            vecInGradQueue->FreeTensor(inGradLocalTensorCompute);

            vecOutQueue->EnQue<tType>(outLocalTensor);
            LocalTensor<tType> copyOutLocalTensor = vecOutQueue->DeQue<tType>();

            DataCopyPad(thisBatchSoftmaxGradGb[offset], copyOutLocalTensor, copyParams);

            vecOutQueue->FreeTensor(copyOutLocalTensor);
            remain = remain - thisLen;
        }
    }

private:
    int softmaxShpeOfOneBatch;
    AttentionFusionGradArgs args;

    TQue<QuePosition::VECIN, 1>* vecInQueue;
    TQue<QuePosition::VECIN, 1>* vecInGradQueue;
    TQue<QuePosition::VECOUT, 1>* vecOutQueue;
    TBuf<TPosition::VECCALC>* tmpBuff;

    GlobalTensor<tType> softmaxOut;
    GlobalTensor<tType> gradSoftmax;
};
}  // namespace AscendFusionGrad
#endif