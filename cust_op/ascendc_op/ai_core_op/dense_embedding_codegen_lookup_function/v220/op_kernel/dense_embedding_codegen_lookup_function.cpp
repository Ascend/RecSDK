/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t TABLE_MAX_NUM = 1024;
constexpr uint32_t MES_LENGTH = 3;
constexpr uint32_t BASIC_PROCESS_UNIT_SIZE = 32;
constexpr uint32_t DATA_COPY_ALIGN_UNIT = 1;

struct IndicesInfo {
    int32_t indicesStart;
    int32_t indicesEnd;
    int32_t weightsOffset;
};

struct InitParams {
    GM_ADDR devWeights;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR hashSizeCumsum;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR indiceWeights;
    GM_ADDR bOffset;
    GM_ADDR vbeOutputOffsetsFeatureRank;
    GM_ADDR vbeBOffsetsRankPerFeature;
    GM_ADDR out;
    GM_ADDR workspace;
    
    int32_t formerCoreNum;
    int32_t formerCoreLength;
    int32_t formerTileNum;
    int32_t formerTileLength;
    int32_t formerLastTileLength;
    int32_t tailCoreNum;
    int32_t tailCoreLength;
    int32_t tailTileNum;
    int32_t tailTileLength;
    int32_t tailLastTileLength;
    int32_t weightsOffsetsLength;
    int32_t batchSize;
    int32_t embedDimLength;
    int32_t indicesAllLength;
    int32_t devWeightsLength;
    int32_t alignedEmbedDimLength;
};

using namespace AscendC;

namespace AscendC {
template <typename T> class DenseEmbeddingCodegenLookupFunction {
public:
    __aicore__ inline DenseEmbeddingCodegenLookupFunction() {}

    __aicore__ inline void Init(const InitParams& params)
    {
        this->blockIdx = GetBlockIdx();

        this->formerCoreNum = params.formerCoreNum;
        this->formerCoreLength = params.formerCoreLength;
        this->formerTileNum = params.formerTileNum;
        this->formerTileLength = params.formerTileLength;
        this->formerLastTileLength = params.formerLastTileLength;

        this->tailCoreNum = params.tailCoreNum;
        this->tailCoreLength = params.tailCoreLength;
        this->tailTileNum = params.tailTileNum;
        this->tailTileLength = params.tailTileLength;
        this->tailLastTileLength = params.tailLastTileLength;

        this->weightsOffsetsLength = params.weightsOffsetsLength;
        this->batchSize = params.batchSize;
        this->embedDimLength = params.embedDimLength;
        this->indicesAllLength = params.indicesAllLength;
        this->devWeightsLength = params.devWeightsLength;
        this->intBlockDim = BASIC_PROCESS_UNIT_SIZE / sizeof(T);
        this->alignedEmbedDimLength = params.alignedEmbedDimLength;

        devWeightsGlobal.SetGlobalBuffer((__gm__ float32_t *)params.devWeights,
            this->devWeightsLength * this->embedDimLength);
        weightsOffsetsGlobal.SetGlobalBuffer((__gm__ T *)params.weightsOffsets, this->weightsOffsetsLength);
        offsetsGlobal.SetGlobalBuffer((__gm__ T *)params.offsets, this->weightsOffsetsLength * this->batchSize + 1);
        outGlobal.SetGlobalBuffer((__gm__ float32_t *)params.out, this->indicesAllLength * this->embedDimLength);
        if (this->blockIdx < this->formerCoreNum) {
            this->tileNum = this->formerTileNum;
            this->tileLength = this->formerTileLength;
            this->lastTileLength = this->formerLastTileLength;
            this->indicesCoreOffset = this->blockIdx * this->formerCoreLength;
            indicesGlobal.SetGlobalBuffer((__gm__ T *)params.indices + this->blockIdx * this->formerCoreLength,
                this->formerCoreLength);
        } else {
            this->tileNum = this->tailTileNum;
            this->tileLength = this->tailTileLength;
            this->lastTileLength = this->tailLastTileLength;
            this->indicesCoreOffset = this->formerCoreNum * this->formerCoreLength +
                (this->blockIdx - this->formerCoreNum) * this->tailCoreLength;
            indicesGlobal.SetGlobalBuffer((__gm__ T *)params.indices + this->formerCoreNum * this->formerCoreLength +
                (this->blockIdx - this->formerCoreNum) * this->tailCoreLength,
                this->tailCoreLength);
        }

        pipe.InitBuffer(inQueueDevWeights, 1, this->alignedEmbedDimLength * sizeof(float32_t));
        pipe.InitBuffer(inQueueIndices, 1,
            (this->formerTileLength + this->intBlockDim - 1) / this->intBlockDim * this->intBlockDim * sizeof(T));
        pipe.InitBuffer(inQueueWeightsOffset, 1,
            (this->weightsOffsetsLength + this->intBlockDim - 1) / this->intBlockDim * this->intBlockDim * sizeof(T));
        pipe.InitBuffer(inQueueOffsets, 1,
            (this->weightsOffsetsLength * this->batchSize + 1 + this->intBlockDim - 1) / this->intBlockDim *
            this->intBlockDim * sizeof(T));
        pipe.InitBuffer(outQueue, 1, this->formerTileLength * this->alignedEmbedDimLength * sizeof(float32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> weightsOffsetsLocal = inQueueWeightsOffset.AllocTensor<T>();
        DataCopy(weightsOffsetsLocal, weightsOffsetsGlobal,
            (this->weightsOffsetsLength + this->intBlockDim - 1) / this->intBlockDim * this->intBlockDim);
        inQueueWeightsOffset.EnQue(weightsOffsetsLocal);
        LocalTensor<T> weightsOffsetProcessLocal = inQueueWeightsOffset.DeQue<T>();
        LocalTensor<T> offsetsLocal = inQueueOffsets.AllocTensor<T>();
        DataCopy(offsetsLocal, offsetsGlobal,
            (this->weightsOffsetsLength * this->batchSize + 1 + this->intBlockDim - 1) / this->intBlockDim *
            this->intBlockDim);
        inQueueOffsets.EnQue(offsetsLocal);
        LocalTensor<T> offsetsProcessLocal = inQueueOffsets.DeQue<T>();

        for (int32_t i = 0; i < this->tileNum; i++) {
            int32_t length = this->tileLength;
            if (i == this->tileNum - 1) {
                length = this->lastTileLength;
            }
            CopyIn(i, length);
            Compute(i, length, weightsOffsetProcessLocal, offsetsProcessLocal);
            CopyOut(i, length);
        }

        inQueueWeightsOffset.FreeTensor(weightsOffsetProcessLocal);
        inQueueOffsets.FreeTensor(offsetsProcessLocal);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t length)
    {
        LocalTensor<T> indicesLocal = inQueueIndices.AllocTensor<T>();
        DataCopy(indicesLocal, indicesGlobal[progress * this->tileLength],
            (length + this->intBlockDim - 1) / this->intBlockDim * this->intBlockDim);
        inQueueIndices.EnQue(indicesLocal);
    }

    __aicore__ inline void Compute(int32_t progress, int32_t length, LocalTensor<T> weightsOffsetsLocal,
        LocalTensor<T> offsetsLocal)
    {
        if (this->weightsOffsetsLength > TABLE_MAX_NUM) {
            return;
        }
        
        this->ResetIndicesArr(this->weightsOffsetsLength);
        
        for (int32_t j = 0; j < this->weightsOffsetsLength; j++) {
            int32_t indicesStart = this->indicesCoreOffset + progress * this->tileLength;
            int32_t offsetsStart = offsetsLocal.GetValue(j * this->batchSize);
            int32_t offsetsEnd = offsetsLocal.GetValue((j + 1) * this->batchSize);
            
            int32_t weightsOffsetsValue = weightsOffsetsLocal.GetValue(j) / this->embedDimLength;
            if (indicesStart >= offsetsStart && indicesStart < offsetsEnd) {
                this->ProcessIndicesRange(j, length, indicesStart, offsetsEnd, weightsOffsetsValue);
                break;
            }
        }

        PipeBarrier<PIPE_ALL>();
        this->ProcessWeightsData(length);
    }

    __aicore__ inline void ProcessWeightsData(int32_t length)
    {
        LocalTensor<T> indicesLocal = inQueueIndices.DeQue<T>();
        LocalTensor<float32_t> outLocal = outQueue.AllocTensor<float32_t>();
        int32_t processCopyNum = 0;
        for (int32_t i = 0; i < this->weightsOffsetsLength && i < TABLE_MAX_NUM; i++) {
            int32_t indicesStart = this->indicesArr[i].indicesStart;
            int32_t indicesEnd = this->indicesArr[i].indicesEnd;
            int32_t weightsOffset = this->indicesArr[i].weightsOffset;
            if (indicesEnd != 0) {
                for (int32_t j = indicesStart; j < indicesEnd; j++) {
                    int32_t indicesNum = indicesLocal.GetValue(j) + weightsOffset;
                    LocalTensor<float32_t> devWeightsLocal = inQueueDevWeights.AllocTensor<float32_t>();
                    this->Gm2UbDataCopyAligned(1, this->embedDimLength, this->embedDimLength,
                        this->alignedEmbedDimLength, devWeightsGlobal[indicesNum * this->embedDimLength],
                        devWeightsLocal);
                    inQueueDevWeights.EnQue(devWeightsLocal);
                    LocalTensor<float32_t> devWeightsProcessLocal = inQueueDevWeights.DeQue<float32_t>();
                    DataCopy(outLocal[processCopyNum * this->alignedEmbedDimLength], devWeightsProcessLocal,
                        this->alignedEmbedDimLength);
                    inQueueDevWeights.FreeTensor(devWeightsProcessLocal);
                    processCopyNum++;
                }
            }
        }
		        
        outQueue.EnQue(outLocal);
        inQueueIndices.FreeTensor(indicesLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t length)
    {
        LocalTensor<float32_t> outLocal = outQueue.DeQue<float32_t>();
        this->Ub2GmDataCopyAligned(length, this->embedDimLength, this->alignedEmbedDimLength, this->embedDimLength,
            outLocal, outGlobal[(this->indicesCoreOffset + progress * this->tileLength) * this->embedDimLength]);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void ResetIndicesArr(int32_t count)
    {
        for (int32_t i = 0; i < count && i < TABLE_MAX_NUM; i++) {
            this->indicesArr[i] = IndicesInfo();
        }
    }

    __aicore__ inline void ProcessIndicesRange(int32_t j, int32_t length, int32_t indicesStart, int32_t offsetsEnd,
                                               int32_t weightsOffsetsValue)
    {
        if (j == this->weightsOffsetsLength - 1 || indicesStart + length < offsetsEnd) {
            this->indicesArr[j].indicesStart = 0;
            this->indicesArr[j].indicesEnd = length;
            this->indicesArr[j].weightsOffset = weightsOffsetsValue;
        } else {
            this->indicesArr[j].indicesStart = 0;
            this->indicesArr[j].indicesEnd = offsetsEnd - indicesStart;
            this->indicesArr[j].weightsOffset = weightsOffsetsValue;
            j = j + 1;
            
            // 添加边界检查防止数组越界
            if (j >= this->weightsOffsetsLength) {
                return;
            }
            
            while (indicesStart + length >= this->offsetsGlobal.GetValue((j + 1) * this->batchSize)) {
                if (j == this->weightsOffsetsLength - 1) {
                    break;
                }
                this->indicesArr[j].indicesStart = this->indicesArr[j - 1].indicesEnd;
                this->indicesArr[j].indicesEnd = this->offsetsGlobal.GetValue((j + 1) * this->batchSize) - indicesStart;
                this->indicesArr[j].weightsOffset = this->weightsOffsetsGlobal.GetValue(j) / this->embedDimLength;
                j = j + 1;
            }
            // 添加边界检查防止数组越界
            if (j < TABLE_MAX_NUM) {
                this->indicesArr[j].indicesStart = this->indicesArr[j - 1].indicesEnd;
                this->indicesArr[j].indicesEnd = length;
                this->indicesArr[j].weightsOffset = this->weightsOffsetsGlobal.GetValue(j) / this->embedDimLength;
            }
        }
    }

    __aicore__ inline void Gm2UbDataCopyAligned(const uint32_t &rows, const uint32_t &cols,
        const uint32_t &src_offset, const uint32_t &dst_offset, const AscendC::GlobalTensor<float32_t> &src_tensor,
        AscendC::LocalTensor<float32_t> &dst_tensor, float32_t pad_val = 0)
    {
        constexpr uint32_t elems_per_block = BASIC_PROCESS_UNIT_SIZE / sizeof(float32_t);

        if ((cols % elems_per_block == 0) && (src_offset % elems_per_block == 0) &&
            (dst_offset % elems_per_block == 0)) {
            AscendC::DataCopyParams gm2ub_datacopy_params;
            gm2ub_datacopy_params.blockCount = rows;
            gm2ub_datacopy_params.blockLen = cols / elems_per_block;
            gm2ub_datacopy_params.srcStride = (src_offset - cols) / elems_per_block;
            gm2ub_datacopy_params.dstStride = (dst_offset - cols) / elems_per_block;
            DataCopy(dst_tensor, src_tensor, gm2ub_datacopy_params);
        } else {
            AscendC::DataCopyExtParams gm2ub_datacopy_params;
            gm2ub_datacopy_params.blockCount = rows;
            gm2ub_datacopy_params.blockLen = cols * sizeof(float32_t);
            gm2ub_datacopy_params.srcStride = (src_offset - cols) * sizeof(float32_t);
            gm2ub_datacopy_params.dstStride = (dst_offset - cols) * sizeof(float32_t);

            AscendC::DataCopyPadExtParams<float32_t> pad_params;
            uint32_t padding_size = (cols + elems_per_block - 1) / elems_per_block * elems_per_block - cols;
            pad_params.isPad = padding_size != 0;
            pad_params.paddingValue = pad_val;
            pad_params.leftPadding = 0;
            pad_params.rightPadding = padding_size;
            DataCopyPad(dst_tensor, src_tensor, gm2ub_datacopy_params, pad_params);
        }
    }

    __aicore__ inline void Ub2GmDataCopyAligned(const uint32_t &rows, const uint32_t &cols,
        const uint32_t &src_offset, const uint32_t &dst_offset, const AscendC::LocalTensor<float32_t> &src_tensor,
        const AscendC::GlobalTensor<float32_t> &dst_tensor)
    {
        constexpr uint32_t aligned_block = BASIC_PROCESS_UNIT_SIZE / sizeof(float32_t);

        if ((cols % aligned_block == 0) && (src_offset % aligned_block == 0) &&
            (dst_offset % aligned_block == 0)) {
            AscendC::DataCopyParams ub2gm_datacopy_params;
            ub2gm_datacopy_params.blockCount = rows;
            ub2gm_datacopy_params.blockLen = cols / aligned_block;
            ub2gm_datacopy_params.srcStride = (src_offset - cols) / aligned_block;
            ub2gm_datacopy_params.dstStride = (dst_offset - cols) / aligned_block;
            DataCopy(dst_tensor, src_tensor, ub2gm_datacopy_params);
        } else {
            AscendC::DataCopyExtParams ub2gm_datacopy_params;
            ub2gm_datacopy_params.blockCount = rows;
            ub2gm_datacopy_params.blockLen = cols * sizeof(float32_t);
            ub2gm_datacopy_params.srcStride = (src_offset - cols) / aligned_block;
            ub2gm_datacopy_params.dstStride = (dst_offset - cols) * sizeof(float32_t);
            ub2gm_datacopy_params.rsv = 0;
            DataCopyPad(dst_tensor, src_tensor, ub2gm_datacopy_params);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueDevWeights, inQueueIndices, inQueueWeightsOffset, inQueueOffsets;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<T> indicesGlobal;
    GlobalTensor<T> weightsOffsetsGlobal;
    GlobalTensor<T> offsetsGlobal;
    GlobalTensor<float32_t> devWeightsGlobal;
    GlobalTensor<float32_t> outGlobal;

    int32_t blockIdx = 0;
    int32_t formerCoreNum;
    int32_t formerCoreLength;
    int32_t formerTileNum;
    int32_t formerTileLength;
    int32_t formerLastTileLength;
    int32_t tailCoreNum;
    int32_t tailCoreLength;
    int32_t tailTileNum;
    int32_t tailTileLength;
    int32_t tailLastTileLength;

    int32_t weightsOffsetsLength;
    int32_t embedDimLength;
    int32_t batchSize;
    int32_t indicesAllLength;
    int32_t devWeightsLength;

    int32_t tileNum;
    int32_t tileLength;
    int32_t lastTileLength;
    int32_t indicesCoreOffset;
    int32_t intBlockDim;
    int32_t alignedEmbedDimLength;
    IndicesInfo indicesArr[TABLE_MAX_NUM] = {};
};
}

extern "C" __global__ __aicore__ void dense_embedding_codegen_lookup_function(GM_ADDR dev_weights,
    GM_ADDR weights_offsets, GM_ADDR D_offsets, GM_ADDR hash_size_cumsum, GM_ADDR indices, GM_ADDR offsets,
    GM_ADDR indice_weights, GM_ADDR B_offset, GM_ADDR vbe_output_offsets_feature_rank,
    GM_ADDR vbe_B_offsets_rank_per_feature, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    DenseEmbeddingCodegenLookupFunction<DTYPE_INDICES> op;
    InitParams params;
    params.devWeights = dev_weights;
    params.weightsOffsets = weights_offsets;
    params.dOffsets = D_offsets;
    params.hashSizeCumsum = hash_size_cumsum;
    params.indices = indices;
    params.offsets = offsets;
    params.indiceWeights = indice_weights;
    params.bOffset = B_offset;
    params.vbeOutputOffsetsFeatureRank = vbe_output_offsets_feature_rank;
    params.vbeBOffsetsRankPerFeature = vbe_B_offsets_rank_per_feature;
    params.out = out;
    params.workspace = workspace;
    params.formerCoreNum = tiling_data.formerCoreNum;
    params.formerCoreLength = tiling_data.formerCoreLength;
    params.formerTileNum = tiling_data.formerTileNum;
    params.formerTileLength = tiling_data.formerTileLength;
    params.formerLastTileLength = tiling_data.formerLastTileLength;
    params.tailCoreNum = tiling_data.tailCoreNum;
    params.tailCoreLength = tiling_data.tailCoreLength;
    params.tailTileNum = tiling_data.tailTileNum;
    params.tailTileLength = tiling_data.tailTileLength;
    params.tailLastTileLength = tiling_data.tailLastTileLength;
    params.weightsOffsetsLength = tiling_data.weightsOffsetsLength;
    params.batchSize = tiling_data.batchSize;
    params.embedDimLength = tiling_data.embedDimLength;
    params.indicesAllLength = tiling_data.indicesAllLength;
    params.devWeightsLength = tiling_data.devWeightsLength;
    params.alignedEmbedDimLength = tiling_data.alignedEmbedDimLength;
    op.Init(params);
    op.Process();
}