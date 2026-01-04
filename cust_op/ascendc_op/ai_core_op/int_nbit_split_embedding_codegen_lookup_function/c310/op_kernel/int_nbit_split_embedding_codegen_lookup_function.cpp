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
#include "common.h"
#include "int_nbit_split_embedding_pooling_kernel.h"
#include "int_nbit_split_embedding_nobag_kernel.h"

// 根据TILING_KEY分发到对应的kernel
template <typename OutputType>
__aicore__ inline void RunByTilingKey(Args& args)
{
    if (TILING_KEY_IS(2)) {
        // Nobag模式
        IntNBitSplitEmbeddingNobag::IntNBitSplitEmbeddingNobagKernel<OutputType> k(args);
        k.Compute();
    } else if (TILING_KEY_IS(0)) {
        // Bag模式，int32_t indices
        IntNBitSplitEmbeddingPooling::IntNBitSplitEmbeddingPoolingKernel<int32_t, OutputType> k(args);
        k.Compute();
    } else if (TILING_KEY_IS(1)) {
        // Bag模式，int64_t indices
        IntNBitSplitEmbeddingPooling::IntNBitSplitEmbeddingPoolingKernel<int64_t, OutputType> k(args);
        k.Compute();
    } else {
        ASCENDC_ASSERT(false, "Unsupported TILING_KEY");
    }
}

// 根据outputDtype分发到对应的输出类型
__aicore__ inline void DispatchOutputDtype(int64_t outputDtype, Args& args)
{
    if (outputDtype == static_cast<int64_t>(SparseType::FP32)) {
        RunByTilingKey<float>(args);
    } else if (outputDtype == static_cast<int64_t>(SparseType::FP16)) {
        RunByTilingKey<half>(args);
    } else if (outputDtype == static_cast<int64_t>(SparseType::BF16)) {
        RunByTilingKey<bfloat16_t>(args);
    } else if (outputDtype == static_cast<int64_t>(SparseType::INT8)) {
        RunByTilingKey<uint8_t>(args);
    } else {
        ASCENDC_ASSERT(false, "Unsupported output dtype");
    }
}

extern "C" __global__ __aicore__ void int_nbit_split_embedding_codegen_lookup_function(
    GM_ADDR devWeights,           // uint8_t: quantized weights (FP8/INT8/INT4/INT2)
    GM_ADDR uvmWeights,           // uint8_t: UVM weights
    GM_ADDR lxuCacheWeights,      // uint8_t: cache weights
    GM_ADDR weightsPlacements,    // int32_t: weight placement type
    GM_ADDR weightsOffsets,       // int64_t: weight offsets for each table
    GM_ADDR weightsTys,           // uint8_t: weight type (SparseType) for each table
    GM_ADDR dOffsets,             // int32_t: dimension offsets for each table
    GM_ADDR indices,              // int32_t/int64_t: indices to lookup
    GM_ADDR offsets,              // int32_t/int64_t: offsets for bagging
    GM_ADDR lxuCacheLocations,    // int32_t: cache locations
    GM_ADDR offsetPerKey,         // int32_t/int64_t: 每张表在offsets中的起始位置
    GM_ADDR indiceWeights,        // float: 新增，可选参数，weighted模式使用
    GM_ADDR out,                  // float: output tensor
    GM_ADDR workspace,            // workspace buffer
    GM_ADDR tiling)               // tiling data
{
    Args args{
        devWeights, uvmWeights, lxuCacheWeights,
        weightsPlacements, weightsOffsets, weightsTys,
        dOffsets, indices, offsets, lxuCacheLocations,
        offsetPerKey, indiceWeights,
        out, tiling, workspace
    };

    // 获取outputDtype（用于确定OutputType模板参数）
    GET_TILING_DATA(tilingData, tiling);
    int64_t outputDtype = tilingData.outputDtype;

    // TILING_KEY: 0=BAG_INT32_KEY, 1=BAG_INT64_KEY, 2=NOBAG_KEY
    DispatchOutputDtype(outputDtype, args);
}
