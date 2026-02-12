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

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_crtp_base.h"
#include "structure/unique_jagged_tensor_input.h"

using namespace AscendC;

namespace BackwardCodegenUnweightedExact {

struct ComputeUniqueArgs {
    int64_t tableIndex;
    int64_t embedDim;
    int64_t inOffset;
    int64_t thisLen;
    int64_t startInd;
};

template <MomentumLayoutType layoutType, typename OptimizerT>
class BackwardCodegenUnweightedExactKernelUnique
    : public BackwardCodegenUnweightedExactCRTPBase<BackwardCodegenUnweightedExactKernelUnique<layoutType, OptimizerT>,
                                                    layoutType, OptimizerT> {
public:
    __aicore__ inline BackwardCodegenUnweightedExactKernelUnique() {}

    __aicore__ inline void Init(Args args)
    {
        BackwardCodegenUnweightedExactCRTPBase<BackwardCodegenUnweightedExactKernelUnique<layoutType, OptimizerT>,
                                               layoutType, OptimizerT>::Init(args);
        InitUnique(args);
    }

    __aicore__ inline void InitUnique(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        this->pipe_.InitBuffer(queIndices, 1, MAX_ARGS_PIPE_LEN * sizeof(int64_t));
        uniqueJaggedInput_.Init(args, tilingData);
    }

    __aicore__ inline void ClearGrad()
    {
        this->workloadSharder_.Compute(uniqueJaggedInput_.GetUniqueIdDim0());
        int64_t loopLen = this->blockLen_ / this->embeddingTable_.GetMaxDim();
        int64_t loops = this->workloadSharder_.length / loopLen;
        int64_t tailLen = this->workloadSharder_.length % loopLen;
        LocalTensor<float> outLt = this->queOut_.template AllocTensor<float>();
        Duplicate<float>(outLt, 0.0, this->blockLen_);
        this->queOut_.template EnQue(outLt);
        LocalTensor<float> newOutLt = this->queOut_.template DeQue<float>();
        for (int64_t i = 0; i < loops; i++) {
            int64_t outOffset = (this->workloadSharder_.start + i * loopLen) * this->embeddingTable_.GetMaxDim();
            CpLocal2Gm(this->gradientFlow_.GetOutputTensor()[outOffset], newOutLt, this->blockLen_);
        }
        if (tailLen > 0) {
            int64_t outOffset = (this->workloadSharder_.start + loops * loopLen) * this->embeddingTable_.GetMaxDim();
            CpLocal2Gm(this->gradientFlow_.GetOutputTensor()[outOffset], newOutLt,
                       tailLen * this->embeddingTable_.GetMaxDim());
        }
        this->queOut_.template FreeTensor(newOutLt);
    }

    __aicore__ inline void ComputeGradBag(ComputeUniqueArgs& args, float meanLen)
    {
        LocalTensor<float> inputLt = this->queIn_.template AllocTensor<float>();
        LocalTensor<float> outputLt = this->queOut_.template AllocTensor<float>();
        LocalTensor<int64_t> indicesLt = queIndices.template AllocTensor<int64_t>();

        CpGm2Local(indicesLt, uniqueJaggedInput_.GetInverseIndicesGT()[args.startInd], args.thisLen);
        int64_t inverseOffset = uniqueJaggedInput_.GetUniqueCountPrefixSum(args.tableIndex);
        CpGm2Local(inputLt, this->gradientFlow_.GetInputTensor()[args.inOffset], args.embedDim);

        queIndices.EnQue(indicesLt);
        this->queIn_.template EnQue(inputLt);

        inputLt = this->queIn_.template DeQue<float>();
        indicesLt = queIndices.template DeQue<int64_t>();

        if (this->gradientFlow_.GetPoolMode() == PoolingMode::POOL_MODE_MEAN) {
            Muls(outputLt, inputLt, meanLen, args.embedDim);
        } else {
            DataCopy(outputLt, inputLt, args.embedDim);
        }

        this->queOut_.template EnQue(outputLt);
        LocalTensor<float> newOutLt = this->queOut_.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < args.thisLen; i++) {
            int64_t outOffset = (indicesLt.GetValue(i) + inverseOffset) * this->embeddingTable_.GetMaxDim();
            CpLocal2Gm(this->gradientFlow_.GetOutputTensor()[outOffset], newOutLt, args.embedDim);
        }
        SetAtomicNone();
        this->queIn_.template FreeTensor(inputLt);
        this->queOut_.template FreeTensor(newOutLt);
        queIndices.template FreeTensor(indicesLt);
    }

    __aicore__ inline void ComputeGradNoBag(ComputeUniqueArgs& args)
    {
        LocalTensor<float> inputLt = this->queIn_.template AllocTensor<float>();
        LocalTensor<float> outputLt = this->queOut_.template AllocTensor<float>();
        LocalTensor<int64_t> indicesLt = queIndices.template AllocTensor<int64_t>();

        CpGm2Local(indicesLt, uniqueJaggedInput_.GetInverseIndicesGT()[args.startInd], args.thisLen);
        int64_t inverseOffset =
            uniqueJaggedInput_.GetUniqueCountPrefixSum(args.tableIndex) * this->embeddingTable_.GetMaxDim();
        CpGm2Local(inputLt, this->gradientFlow_.GetInputTensor()[args.inOffset],
                   this->embeddingTable_.GetMaxDim() * args.thisLen);

        queIndices.EnQue(indicesLt);
        this->queIn_.template EnQue(inputLt);
        inputLt = this->queIn_.template DeQue<float>();
        indicesLt = queIndices.template DeQue<int64_t>();

        DataCopy(outputLt, inputLt, this->embeddingTable_.GetMaxDim() * args.thisLen);
        this->queOut_.template EnQue(outputLt);
        LocalTensor<float> newOutLt = this->queOut_.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < args.thisLen; i++) {
            int64_t outOffset = indicesLt.GetValue(i) * this->embeddingTable_.GetMaxDim() + inverseOffset;
            CpLocal2Gm(this->gradientFlow_.GetOutputTensor()[outOffset],
                       newOutLt[i * this->embeddingTable_.GetMaxDim()], args.embedDim);
        }
        SetAtomicNone();
        this->queIn_.template FreeTensor(inputLt);
        this->queOut_.template FreeTensor(newOutLt);
        queIndices.template FreeTensor(indicesLt);
    }

    __aicore__ inline void ComputeGrad()
    {
        if (this->gradientFlow_.GetPoolMode() == PoolingMode::POOL_MODE_NONE) {
            ComputeGradEC();
        } else {
            ComputeGradEBC();
        }
    }

    __aicore__ inline void ComputeGradEC()
    {
        int64_t indicesNumOneBlock = this->ComputeIndicesNumOneBlock(1);
        int64_t lastIndices = 0;
        int64_t thisLen = 0;
        for (int64_t i = 1; i <= this->embeddingTable_.GetWeightsOffsetsDim0(); i++) {
            int64_t rawCount = uniqueJaggedInput_.GetRawCount(i);
            this->workloadSharder_.Compute(rawCount - lastIndices);
            thisLen = this->workloadSharder_.length;
            int64_t startIndices = this->workloadSharder_.start + lastIndices; // 上一张表的偏移+table_i的偏移
            lastIndices = rawCount;
            if (thisLen <= 0) {
                continue;
            }
            int32_t remain = thisLen;
            int64_t thisOffsetIndex = startIndices;

            // datacopy In params
            int64_t tableIndex = i - 1;
            int64_t embedDim = this->embeddingTable_.GetEmbedDim(tableIndex);
            int64_t inputOffset = startIndices * this->gradientFlow_.GetInputDim1();
            while (remain > 0) {
                if (thisLen > indicesNumOneBlock) {
                    thisLen = indicesNumOneBlock;
                }
                remain -= thisLen;
                ComputeUniqueArgs args{tableIndex, embedDim, inputOffset, thisLen, startIndices};
                ComputeGradNoBag(args);
                inputOffset += thisLen * this->gradientFlow_.GetInputDim1();
                startIndices += thisLen;
                thisLen = remain;
            }
        }
    }

    __aicore__ inline void ComputeGradEBC()
    {
        this->workloadSharder_.Compute(this->jaggedInput_.GetOffsetsSize() - 1);
        if (this->workloadSharder_.length == 0) {
            return;
        }
        int64_t indicesNumOneBlock = this->ComputeIndicesNumOneBlock(1);

        int64_t batchs = (this->jaggedInput_.GetOffsetsDimSize() - 1) / this->embeddingTable_.GetWeightsOffsetsDim0();
        for (int64_t loop = 0; loop < this->workloadSharder_.length; loop++) {
            int64_t i = (this->workloadSharder_.start + loop) / this->embeddingTable_.GetWeightsOffsetsDim0();
            int64_t j = (this->workloadSharder_.start + loop) % this->embeddingTable_.GetWeightsOffsetsDim0();
            int64_t thisOffsetIndex = j * batchs + i;
            int64_t startIndices = this->jaggedInput_.GetOffset(thisOffsetIndex);
            int64_t endIndices = this->jaggedInput_.GetOffset(thisOffsetIndex + 1);
            int32_t thisLen = endIndices - startIndices;

            if (thisLen <= 0) {
                continue;
            }

            int32_t remain = thisLen;
            float meanLen = 1 / static_cast<float>(thisLen);

            // dataCopy In params
            int64_t tableIndex = thisOffsetIndex / batchs;
            int64_t inputBatchInd = thisOffsetIndex % batchs;

            int64_t embedDim;
            int64_t inputEmbedOffset;
            this->embeddingTable_.GetDimAndOffset(tableIndex, embedDim, inputEmbedOffset);
            int64_t inputOffset = inputBatchInd * this->gradientFlow_.GetInputDim1() + inputEmbedOffset;
            while (remain > 0) {
                if (thisLen > indicesNumOneBlock) {
                    thisLen = indicesNumOneBlock;
                }
                remain -= thisLen;
                ComputeUniqueArgs args{tableIndex, embedDim, inputOffset, thisLen, startIndices};
                ComputeGradBag(args, meanLen);
                startIndices += thisLen;
                thisLen = remain;
            }
        }
    }

    // 实现CRTP接口
    __aicore__ inline void ComputeImpl(Args args)
    {
        Init(args);
        ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
    }

    // 权重更新调度方法
    __aicore__ inline void UpdateWeightsSchedulerImpl(Args args)
    {
        this->InitCommonVariables(OutputCount<layoutType>::value);
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < uniqueJaggedInput_.GetUniqueHashIdCount(); i++) {
            int64_t index = uniqueJaggedInput_.GetUniqueCountPrefixSum(i);
            if (index != lastIndices) { // 每张表上的indices尽量均分到每张卡上
                this->workloadSharder_.Compute(index - lastIndices);
                if (this->workloadSharder_.length > 0) {
                    this->tableIndex_ = i - 1;
                    this->thisTableOffset_ = this->workloadSharder_.start + lastIndices;
                    UpdateEmbed();
                }
                lastIndices = index;
            }
        }
    }

    // 权重更新方法
    __aicore__ inline void UpdateEmbed()
    {
        this->indicesNumOneBlock_ = this->ComputeIndicesNumOneBlock(this->numOfOut_);
        int64_t thisLen = this->workloadSharder_.length;
        int64_t remain = this->workloadSharder_.length;

        int64_t embedDim = this->embeddingTable_.GetEmbedDim(this->tableIndex_);

        while (remain > 0) {
            if (remain > this->indicesNumOneBlock_) {
                thisLen = this->indicesNumOneBlock_;
            }

            int calcLen = thisLen * this->embeddingTable_.GetMaxDim();

            remain -= thisLen;
            LocalTensor<float> inputLt = this->queIn_.template AllocTensor<float>();
            LocalTensor<float> outputLt = this->queOut_.template AllocTensor<float>();

            // copyIn
            CpGm2Local(inputLt,
                       this->gradientFlow_.GetOutputTensor()[this->thisTableOffset_ *
                       this->embeddingTable_.GetMaxDim()], calcLen);
            this->queIn_.template EnQue(inputLt);
            // CopyIn
            int64_t updateArgs[MAX_ARGS_PIPE_LEN];
            CopyInNormal(updateArgs, thisLen, embedDim);
            // compute
            inputLt = this->queIn_.template DeQue<float>();

            ComputeOptimizer(inputLt, outputLt, calcLen);
            this->queOut_.template EnQue(outputLt);

            // copyOut
            CopyOutNormal(updateArgs, thisLen, embedDim);

            this->queIn_.template FreeTensor(inputLt);
            this->thisTableOffset_ += thisLen;
            thisLen = remain;
        }
    }

    // 通用的CopyInNormal实现，使用模板参数控制不同优化器的行为
    __aicore__ inline void CopyInNormal(int64_t* updateArgs, int64_t cnt, int64_t embedDim)
    {
        LocalTensor<float> inputLt = this->queIn_.template DeQue<float>();

        // 在Adagrad的情况下，动量的起始索引是 cnt * this->embeddingTable_.GetMaxDim() * MOMENTUM1_IDX
        int64_t baseMomentIndex = cnt * this->embeddingTable_.GetMaxDim();

        for (int64_t i = 0; i < cnt; i++) {
            // 获取unique索引和权重偏移
            int64_t thisIndForThisTable = uniqueJaggedInput_.GetUniqueIdx(this->thisTableOffset_ + i);
            int64_t thisWeightOffset = this->embeddingTable_.GetWeightOffsetTableValue(this->tableIndex_);
            updateArgs[i] = thisWeightOffset + thisIndForThisTable * embedDim;

            // 根据内存布局类型进行数据复制
            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_ONLY) {
                // Adagrad 需要复制momentum1
                int64_t thisMoment1Index =
                    baseMomentIndex * static_cast<int64_t>(OptimizerState::Index::MOMENTUM1_IDX) +
                    i * this->embeddingTable_.GetMaxDim();

                DataCopy(inputLt[thisMoment1Index], this->optimizerState_.GetMomentum1InGT()[updateArgs[i]], embedDim);
            }

            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM) {
                // ADAM版本需要额外的momentum2复制
                int64_t moment2Index = baseMomentIndex * static_cast<int64_t>(OptimizerState::Index::MOMENTUM2_IDX) +
                                       i * this->embeddingTable_.GetMaxDim();
                DataCopy(inputLt[moment2Index], this->optimizerState_.GetMomentum2InGT()[updateArgs[i]], embedDim);
            }
        }

        this->queIn_.EnQue(inputLt);
    }

    // 通用的CopyOutNormal实现，同样使用模板参数控制不同优化器的行为
    __aicore__ inline void CopyOutNormal(int64_t* outOffset, int64_t cnt, int64_t embedDim)
    {
        LocalTensor<float> newOutLt = this->queOut_.template DeQue<float>();

        SetAtomicAdd<float>();
        // 首先处理权重更新
        for (int64_t i = 0; i < cnt; i++) {
            DataCopy(this->optimizerState_.GetWeightsDevOutGT()[outOffset[i]],
                     newOutLt[i * this->embeddingTable_.GetMaxDim()], embedDim);

            if constexpr (OutputCount<layoutType>::value == EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM) {
                // Adagrad 使用常规momentum1复制
                int64_t thisMoment1Index = cnt * this->embeddingTable_.GetMaxDim() *
                                           static_cast<int64_t>(OptimizerState::Index::MOMENTUM1_IDX);
                DataCopy(this->optimizerState_.GetMomentum1OutGT()[outOffset[i]],
                         newOutLt[thisMoment1Index + i * this->embeddingTable_.GetMaxDim()], embedDim);
            }
        }
        SetAtomicNone();

        // 然后根据内存布局类型复制momentum数据
        for (int64_t i = 0; i < cnt; i++) {
            if constexpr (OutputCount<layoutType>::value ==
                          EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_MOMENTUM2) {
                // Adam版本需要额外的momentum2输出
                int64_t thisMoment1Index = cnt * this->embeddingTable_.GetMaxDim() *
                                           static_cast<int64_t>(OptimizerState::Index::MOMENTUM1_IDX);
                DataCopy(this->optimizerState_.GetMomentum1OutGT()[outOffset[i]],
                         newOutLt[thisMoment1Index + i * this->embeddingTable_.GetMaxDim()], embedDim);
                int64_t thisMoment2Index = cnt * this->embeddingTable_.GetMaxDim() *
                                           static_cast<int64_t>(OptimizerState::Index::MOMENTUM2_IDX);

                DataCopy(this->optimizerState_.GetMomentum2OutGT()[outOffset[i]],
                         newOutLt[thisMoment2Index + i * this->embeddingTable_.GetMaxDim()], embedDim);
            }
        }

        this->queOut_.FreeTensor(newOutLt);
    }

    // 优化器计算实现
    __aicore__ inline void ComputeOptimizer(LocalTensor<float> newInputLt, LocalTensor<float> outLt, int64_t totalLen)
    {
        // 根据内存布局类型和优化器类型决定调用优化器的不同方法
        if constexpr (layoutType == MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_PER_DIM) {
            // Adagrad优化器需要6个参数
            this->optimizer_.Compute(newInputLt, outLt, 0, totalLen, totalLen, this->optimizerConfig_);
        } else if constexpr (layoutType == MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_MOMENTUM2) {
            // Adam优化器需要7个参数
            this->optimizer_.Compute(newInputLt, outLt, 0, totalLen,
                static_cast<int64_t>(OptimizerState::Index::MOMENTUM2_IDX) * totalLen,
                totalLen, this->optimizerConfig_);
        } else if constexpr (layoutType == MomentumLayoutType::LAYOUT_GRAD_ONLY) {
            // SGD优化器需要5个参数
            this->optimizer_.Compute(newInputLt, outLt, 0, totalLen, this->optimizerConfig_);
        }
    }

    UniqueJaggedTensorInput uniqueJaggedInput_;
    TQue<TPosition::VECIN, 1> queIndices;
};

} // namespace BackwardCodegenUnweightedExact
#endif