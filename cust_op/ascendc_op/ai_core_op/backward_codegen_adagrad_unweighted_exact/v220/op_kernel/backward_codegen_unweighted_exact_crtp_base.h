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

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_BASE_FUN_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_BASE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "../../../../common/kernel_common_utils.h"
#include "../../../../common/workload_sharder.h"

#include "structure/args_struct.h"
#include "structure/embedding_table_layout.h"
#include "structure/gradient_flow.h"
#include "structure/jagged_tensor_input.h"
#include "structure/optimizer_config.h"
#include "structure/optimizer_state.h"
#include "structure/pooling_mode_enum.h"

#include "structure/optimizer_layout.h"
#include "optimizers/optimizer_interface.h"

using namespace AscendC;

namespace BackwardCodegenUnweightedExact {

constexpr int USE_QUEUE_NUM = 2;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int DATA_ALIGN_BYTES = 32;

constexpr int8_t NEED_UPDATE = 33;
constexpr int MAX_ARGS_PIPE_LEN = 300;
constexpr int FLAG_LEN = DATA_ALIGN_BYTES / sizeof(int8_t);

// CRTP基类定义
template <typename Derived, MomentumLayoutType layoutType, typename OptimizerT>
class BackwardCodegenUnweightedExactCRTPBase {
public:
    __aicore__ inline BackwardCodegenUnweightedExactCRTPBase() {}

    __aicore__ inline void InitUb(BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        // ub
        blockLen_ = tilingData.ubCanUsed / USE_QUEUE_NUM / sizeof(float);
        blockLen_ = blockLen_ / FLOAT_ALIGNMENT * FLOAT_ALIGNMENT;
    }

    __aicore__ inline void InitPipe()
    {
        // Init pipe_
        pipe_.InitBuffer(queIn_, 1, blockLen_ * sizeof(float));
        pipe_.InitBuffer(queOut_, 1, blockLen_ * sizeof(float));
    }

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        InitUb(tilingData);
        embeddingTable_.Init(args, tilingData);
        jaggedInput_.Init(args, tilingData);
        optimizerState_.Init(args, tilingData);
        optimizerConfig_.Init(args, tilingData);
        gradientFlow_.Init(args, tilingData);

        InitPipe();
    }

    __aicore__ inline void InitCommonVariables(int numOutputs)
    {
        numOfOut_ = numOutputs;
        indicesNumOneBlock_ = ComputeIndicesNumOneBlock(numOfOut_);
    }

    template <typename T> __aicore__ inline void ClearGT(const GlobalTensor<T>& clearGt, int64_t clearSize)
    {
        workloadSharder_.Compute(clearSize);

        int64_t total = workloadSharder_.length;
        int64_t remain = total;
        int thisAlignment = DATA_ALIGN_BYTES / sizeof(T);
        LocalTensor<T> outLt = queOut_.AllocTensor<T>();
        LocalTensor<int32_t> clearLt = outLt.template ReinterpretCast<int32_t>();
        Duplicate<int32_t>(clearLt, (int32_t) 0, blockLen_);
        queOut_.EnQue(outLt);
        LocalTensor<T> newOutLt = queOut_.DeQue<T>();
        while (remain > 0) {
            int64_t thisLen = blockLen_;
            if (remain < thisLen) {
                thisLen = (remain + thisAlignment - 1);
            }
            thisLen = thisLen / thisAlignment * thisAlignment;
            int thisOffset = total - remain;
            DataCopy(clearGt[workloadSharder_.start + thisOffset], newOutLt, thisLen);
            remain = remain - thisLen;
        }
        queOut_.FreeTensor(newOutLt);
    }

    __aicore__ inline int ComputeIndicesNumOneBlock(int64_t numOfOut)
    {
        ASSERT(numOfOut > 0);
        int indicesNum = blockLen_ / numOfOut / embeddingTable_.GetMaxDim();
        if (indicesNum >= MAX_ARGS_PIPE_LEN) {
            indicesNum = MAX_ARGS_PIPE_LEN;
        }
        return indicesNum;
    }

    // 通过CRTP调用派生类实现
    __aicore__ inline void Compute(Args args) { static_cast<Derived*>(this)->ComputeImpl(args); }

    __aicore__ inline void UpdateWeightsScheduler(Args args)
    {
        static_cast<Derived*>(this)->UpdateWeightsSchedulerImpl(args);
    }

    // compute weights update
    int numOfOut_;
    int indicesNumOneBlock_;
    int64_t thisTableOffset_;
    int64_t tableIndex_;

    WorkloadSharder workloadSharder_;

    // Ub
    int64_t blockLen_;
    // Tpipe
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> queIn_;
    TQue<TPosition::VECOUT, 1> queOut_;

    EmbeddingTableLayout embeddingTable_;
    JaggedTensorInput jaggedInput_;
    OptimizerState optimizerState_;
    OptimizerConfig optimizerConfig_;
    GradientFlow gradientFlow_;

protected:
    OptimizerT optimizer_;
};

} // namespace BackwardCodegenUnweightedExact

#endif