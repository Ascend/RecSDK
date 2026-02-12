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

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_crtp_base.h"

using namespace AscendC;
namespace BackwardCodegenUnweightedExact {

struct ComputeArgs {
    int64_t offsetIndex;
    int64_t embedDim;
    int64_t inputOffset;
    int64_t indWeightOffset;
};

struct UpdateArgs {
    int64_t inputOffset;
    int64_t embedDim;
    int64_t thisOutOffset;
    int64_t thisMomentumOffset;
};

// 继承自CRTP基类
template <MomentumLayoutType layoutType, typename OptimizerT>
class BackwardCodegenUnweightedExactKernel
    : public BackwardCodegenUnweightedExactCRTPBase<BackwardCodegenUnweightedExactKernel<layoutType, OptimizerT>,
                                                    layoutType, OptimizerT> {
public:
    __aicore__ inline BackwardCodegenUnweightedExactKernel() {}

    __aicore__ inline void InitPipe() { this->pipe_.InitBuffer(this->queFlagOut, 1, DATA_ALIGN_BYTES); }

    __aicore__ inline void Init(Args args)
    {
        BackwardCodegenUnweightedExactCRTPBase<BackwardCodegenUnweightedExactKernel<layoutType, OptimizerT>, layoutType,
                                               OptimizerT>::Init(args);
        InitPipe();
        updateMaskGT_.SetGlobalBuffer((__gm__ int8_t*) args.workspace, this->embeddingTable_.GetTotalHashSize());
    }

    // 统一处理函数，根据模板参数区分是清零还是计算梯度
    template <bool isClearGrad> __aicore__ inline void ProcessGrad()
    {
        LocalTensor<int8_t> newFlagOutLt;

        if constexpr (isClearGrad) {
            this->workloadSharder_.Compute(this->jaggedInput_.GetIndicesDimSize());
        } else {
            // ComputeGrad的flag初始化部分
            LocalTensor<int8_t> flagOutLt = this->queFlagOut.template AllocTensor<int8_t>();
            LocalTensor<int32_t> clearLt = flagOutLt.ReinterpretCast<int32_t>();
            Duplicate<int32_t>(clearLt, (int32_t) 0, FLAG_LEN / sizeof(int32_t));
            flagOutLt.SetValue(0, NEED_UPDATE);
            this->queFlagOut.EnQue(flagOutLt);
            newFlagOutLt = this->queFlagOut.template DeQue<int8_t>();
        }

        int64_t thisOffsetIndex = this->jaggedInput_.LocateOffsetIndex(this->workloadSharder_.start);

        int64_t total = this->workloadSharder_.length;
        int64_t remain = total;
        int64_t indicesNumOneBlock = this->ComputeIndicesNumOneBlock(1);
        ComputeArgs argsArry[MAX_ARGS_PIPE_LEN];

        while (remain > 0) {
            int64_t thisLen = 0;
            // 调用专门的函数来填充参数数组
            FillComputeArgs<isClearGrad>(thisLen, argsArry, indicesNumOneBlock, remain, total, thisOffsetIndex,
                                         newFlagOutLt);

            // 根据模板参数调用不同的处理函数
            if constexpr (isClearGrad) {
                ProcessClearGrad(thisLen, argsArry);
            } else {
                ProcessComputeGrad(thisLen, argsArry, newFlagOutLt);
            }
        }
    }

    // 填充计算参数的函数
    template <bool isClearGrad>
    __aicore__ inline void FillComputeArgs(int64_t& thisLen, ComputeArgs* argsArry, int64_t indicesNumOneBlock,
                                           int64_t& remain, int64_t total, int64_t& thisOffsetIndex,
                                           LocalTensor<int8_t>& newFlagOutLt)
    {
        while (thisLen < indicesNumOneBlock && remain > 0) {
            int64_t indicesInd = this->workloadSharder_.end - remain;
            remain = remain - 1;
            while (!this->jaggedInput_.IsInOffsetRange(indicesInd, thisOffsetIndex)) {
                thisOffsetIndex = thisOffsetIndex + 1;
            }

            // Which Table Used, and the table embedDim
            int64_t batchSize = this->jaggedInput_.GetBatchSize();
            int64_t tableIndex = thisOffsetIndex / batchSize;

            int64_t embedDim, thisWeightOffset, inputEmbedOffset;
            this->embeddingTable_.GetTableInfo(tableIndex, embedDim, thisWeightOffset);
            inputEmbedOffset = this->embeddingTable_.GetEmbeddingDimOffset(tableIndex);

            int64_t thisIndForThisTable = this->jaggedInput_.GetId(indicesInd);

            // Out offset
            int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;

            int64_t inputBatchInd, inputEmbedOffsetForCalc, inputOffset;

            if constexpr (isClearGrad) {
                // --- ClearGrad 路径 ---
                inputBatchInd = thisOffsetIndex % batchSize;
                inputEmbedOffsetForCalc = inputEmbedOffset;
                inputOffset = inputBatchInd * this->gradientFlow_.GetInputDim1() + inputEmbedOffsetForCalc;

                // ComputeGrad 的原子操作部分在此不执行（自然跳过）
            } else {
                // --- ComputeGrad 路径 ---
                inputBatchInd = thisOffsetIndex % this->gradientFlow_.GetInputDim0();
                inputEmbedOffsetForCalc = this->embeddingTable_.GetEmbeddingDimOffset(tableIndex);

                // 根据 PoolMode 决定 inputOffset
                if (this->gradientFlow_.GetPoolMode() == PoolingMode::POOL_MODE_NONE) {
                    inputOffset = indicesInd * this->gradientFlow_.GetInputDim1();
                } else {
                    inputOffset = inputBatchInd * this->gradientFlow_.GetInputDim1() + inputEmbedOffsetForCalc;
                }

                // ComputeGrad 特有：写 flag
                int64_t thisIndForTotalTable =
                    this->embeddingTable_.GetTableSizePrefixSum(tableIndex) + thisIndForThisTable;
                SetAtomicMax<int8_t>();
                DataCopy(this->updateMaskGT_[thisIndForTotalTable], newFlagOutLt, FLAG_LEN);
                SetAtomicNone();
            }

            argsArry[thisLen] = {thisOffsetIndex, embedDim, inputOffset, thisOutOffset};

            thisLen += 1;
        }
    }

    // ClearGrad的输出处理
    __aicore__ inline void ProcessClearGrad(int64_t thisLen, ComputeArgs* argsArry)
    {
        LocalTensor<float> outLt = this->queOut_.template AllocTensor<float>();
        Duplicate<float>(outLt, 0.0, this->blockLen_);
        this->queOut_.EnQue(outLt);
        LocalTensor<float> newOutLt = this->queOut_.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            ComputeArgs theArgs = argsArry[i];
            CpLocal2Gm(this->gradientFlow_.GetOutputTensor()[theArgs.indWeightOffset],
                       newOutLt[i * this->embeddingTable_.GetMaxDim()], theArgs.embedDim);
        }
        this->queOut_.FreeTensor(newOutLt);
    }

    // ComputeGrad的输入和输出处理
    __aicore__ inline void ProcessComputeGrad(int64_t thisLen, ComputeArgs* argsArry, LocalTensor<int8_t>& newFlagOutLt)
    {
        // ComputeGrad的输入处理
        LocalTensor<float> inputLt = this->queIn_.template AllocTensor<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            ComputeArgs theArgs = argsArry[i];
            CpGm2Local(inputLt[i * this->embeddingTable_.GetMaxDim()],
                       this->gradientFlow_.GetInputTensor()[theArgs.inputOffset], theArgs.embedDim);
        }
        this->queIn_.EnQue(inputLt);

        LocalTensor<float> newInputLt = this->queIn_.template DeQue<float>();
        LocalTensor<float> outLt = this->queOut_.template AllocTensor<float>();

        if (this->gradientFlow_.GetPoolMode() == PoolingMode::POOL_MODE_MEAN) {
            for (int64_t i = 0; i < thisLen; i++) {
                ComputeArgs theArgs = argsArry[i];
                float meanLen = (float) 1 / this->jaggedInput_.GetBagLength(theArgs.offsetIndex);
                Muls<float>(outLt[i * this->embeddingTable_.GetMaxDim()],
                            newInputLt[i * this->embeddingTable_.GetMaxDim()], meanLen,
                            this->embeddingTable_.GetMaxDim());
            }
        } else {
            DataCopy(outLt, newInputLt, this->blockLen_);
        }

        this->queOut_.EnQue(outLt);
        this->queIn_.FreeTensor(newInputLt);

        LocalTensor<float> newOutLt = this->queOut_.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            ComputeArgs theArgs = argsArry[i];
            CpLocal2Gm(this->gradientFlow_.GetOutputTensor()[theArgs.indWeightOffset],
                       newOutLt[i * this->embeddingTable_.GetMaxDim()], theArgs.embedDim);
        }
        SetAtomicNone();
        this->queOut_.FreeTensor(newOutLt);
    }

    __aicore__ inline void ClearGrad() { ProcessGrad<true>(); }

    __aicore__ inline void ComputeGrad() { ProcessGrad<false>(); }

    // 统一的FillUpdateArgs实现，根据布局类型内部特化
    __aicore__ inline int64_t FillUpdateArgs(UpdateArgs* updateArgs, int64_t& remain)
    {
        int64_t cnt = 0;
        while (cnt < this->indicesNumOneBlock_ && remain > 0) {
            int64_t thisIndForTotalTable = this->workloadSharder_.end - remain;
            remain = remain - 1;
            if (thisIndForTotalTable >= this->embeddingTable_.GetTableSizePrefixSum(this->tableIndex_ + 1)) {
                this->tableIndex_ = this->tableIndex_ + 1;
            }

            if (this->updateMaskGT_.GetValue(thisIndForTotalTable) != NEED_UPDATE) {
                continue;
            }

            int64_t thisIndForThisTable = this->embeddingTable_.GetLocalId(thisIndForTotalTable, this->tableIndex_);

            int64_t embedDim;
            int64_t thisWeightOffset;
            this->embeddingTable_.GetTableInfo(this->tableIndex_, embedDim, thisWeightOffset);
            int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;

            updateArgs[cnt].embedDim = embedDim;
            updateArgs[cnt].thisOutOffset = thisOutOffset;

            // 根据布局类型特化处理
            if constexpr (layoutType == MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_ROWWISE) {
                updateArgs[cnt].thisMomentumOffset = thisIndForTotalTable * EngineLayout::MOMENTUM_PAD_NUM;
            }

            cnt += 1;
        }
        return cnt;
    }

    __aicore__ inline void TillingOptimizer()
    {
        this->workloadSharder_.Compute(this->updateMaskGT_.GetSize());
        this->tableIndex_ = this->embeddingTable_.LocateTableIndex(this->workloadSharder_.start);
    }

    // 初始化优化器相关变量
    __aicore__ inline void InitOptimizer(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        this->InitCommonVariables(OutputCount<layoutType>::value); // 初始化公共变量，输出个数由模板参数决定
    }

    // 通用的DataCopyIn实现，使用模板参数控制不同优化器的行为
    __aicore__ inline void DataCopyIn(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> inputLt = this->queIn_.template AllocTensor<float>();

        // 提取循环外不变的索引偏移量计算
        const int64_t gradIndexOffset =
            static_cast<int64_t>(OptimizerState::Index::GRAD_IDX) * this->embeddingTable_.GetMaxDim();
        const int64_t moment1IndexOffset =
            static_cast<int64_t>(OptimizerState::Index::MOMENTUM1_IDX) * this->embeddingTable_.GetMaxDim();
        const int64_t moment2IndexOffset =
            static_cast<int64_t>(OptimizerState::Index::MOMENTUM2_IDX) * this->embeddingTable_.GetMaxDim();
        const int64_t indexMultiplier = this->embeddingTable_.GetMaxDim() * this->numOfOut_;
        const int64_t gradBaseIndex = gradIndexOffset; // 梯度的基本索引

        // 在循环中根据outputType处理不同的momentum需求
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs args = updateArgs[i];
            int64_t gradIndex = i * indexMultiplier + gradIndexOffset;

            // 统一处理梯度复制
            DataCopy(inputLt[i * this->embeddingTable_.GetMaxDim() * this->numOfOut_ + gradBaseIndex],
                     this->gradientFlow_.GetOutputTensor()[args.thisOutOffset], args.embedDim);

            // 根据内存布局类型进行额外的数据复制
            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_ONLY) {
                // Adagrad 和 Rowwise Adagrad 需要复制momentum1
                int64_t moment1Index = i * indexMultiplier + moment1IndexOffset;

                if constexpr (layoutType == MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_ROWWISE) {
                    // Rowwise版本使用thisMomentumOffset和MOMENTUM_PAD_NUM
                    DataCopy(inputLt[moment1Index], this->optimizerState_.GetMomentum1InGT()[args.thisMomentumOffset],
                             static_cast<int64_t>(EngineLayout::MOMENTUM_PAD_NUM));
                } else {
                    // Adagrad 使用args.embedDim
                    DataCopy(inputLt[moment1Index], this->optimizerState_.GetMomentum1InGT()[args.thisOutOffset],
                             args.embedDim);
                }
            }

            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM) {
                // ADAM版本需要额外的momentum2复制
                int64_t moment2Index = i * indexMultiplier + moment2IndexOffset;

                DataCopy(inputLt[moment2Index], this->optimizerState_.GetMomentum2InGT()[args.thisOutOffset],
                         args.embedDim);
            }
        }

        this->queIn_.EnQue(inputLt);
    }

    // 优化器计算实现
    __aicore__ inline void ComputeOptimizer(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> inputLt = this->queIn_.template DeQue<float>();
        LocalTensor<float> outLt = this->queOut_.template AllocTensor<float>();

        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t gradOffset =
                i * this->embeddingTable_.GetMaxDim() * this->numOfOut_ +
                static_cast<int64_t>(OptimizerState::Index::GRAD_IDX) * this->embeddingTable_.GetMaxDim();
            int64_t moment1Offset =
                i * this->embeddingTable_.GetMaxDim() * this->numOfOut_ +
                static_cast<int64_t>(OptimizerState::Index::MOMENTUM1_IDX) * this->embeddingTable_.GetMaxDim();

            // 根据内存布局类型决定是否需要第三个偏移量
            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM) {
                int64_t moment2Offset =
                    i * this->embeddingTable_.GetMaxDim() * this->numOfOut_ +
                    static_cast<int64_t>(OptimizerState::Index::MOMENTUM2_IDX) * this->embeddingTable_.GetMaxDim();

                // 三参数优化器计算（例如Adam）
                this->optimizer_.Compute(inputLt, outLt, gradOffset, moment1Offset, moment2Offset, theArgs.embedDim,
                                         this->optimizerConfig_);
            } else if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_ONLY) {
                // 两参数优化器计算（例如Adagrad）
                this->optimizer_.Compute(inputLt, outLt, gradOffset, moment1Offset, theArgs.embedDim,
                                         this->optimizerConfig_);
            } else {
                // 单参数优化器计算（例如SGD）
                this->optimizer_.Compute(inputLt, outLt, gradOffset, theArgs.embedDim, this->optimizerConfig_);
            }
        }

        this->queOut_.EnQue(outLt);
        this->queIn_.FreeTensor(inputLt);
    }

    // 通用的DataCopyOut实现，同样使用模板参数控制不同优化器的行为
    __aicore__ inline void DataCopyOut(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> outLt = this->queOut_.template DeQue<float>();

        // 提取循环外不变的索引偏移量计算
        const int64_t gradIndexOffset =
            static_cast<int64_t>(OptimizerState::Index::GRAD_IDX) * this->embeddingTable_.GetMaxDim();
        const int64_t moment1IndexOffset =
            static_cast<int64_t>(OptimizerState::Index::MOMENTUM1_IDX) * this->embeddingTable_.GetMaxDim();
        const int64_t moment2IndexOffset =
            static_cast<int64_t>(OptimizerState::Index::MOMENTUM2_IDX) * this->embeddingTable_.GetMaxDim();
        const int64_t indexMultiplier = this->embeddingTable_.GetMaxDim() * this->numOfOut_;

        SetAtomicAdd<float>();
        // 首先处理权重更新
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisGradIndex = i * indexMultiplier + gradIndexOffset;
            DataCopy(this->optimizerState_.GetWeightsDevOutGT()[theArgs.thisOutOffset], outLt[thisGradIndex],
                     theArgs.embedDim);
        }
        SetAtomicNone();

        // 然后根据内存布局类型复制momentum数据
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];

            // 根据内存布局类型进行额外的momentum数据复制
            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_ONLY) {
                // Adagrad 和 Rowwise 都需要复制momentum1
                int64_t thisMoment1Index = i * indexMultiplier + moment1IndexOffset;

                if constexpr (layoutType == MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_ROWWISE) {
                    // Rowwise版本使用thisMomentumOffset且固定长度为MOMENTUM_PAD_NUM
                    DataCopy(this->optimizerState_.GetMomentum1OutGT()[theArgs.thisMomentumOffset],
                             outLt[thisMoment1Index], static_cast<int64_t>(EngineLayout::MOMENTUM_PAD_NUM));
                } else {
                    // Adagrad 使用thisOutOffset
                    DataCopy(this->optimizerState_.GetMomentum1OutGT()[theArgs.thisOutOffset], outLt[thisMoment1Index],
                             theArgs.embedDim);
                }
            }

            if constexpr (OutputCount<layoutType>::value > EngineLayout::OUTPUT_COUNT_LAYOUT_GRAD_MOMENTUM1_PER_DIM) {
                // Adam版本需要额外的momentum2输出
                int64_t thisMoment2Index = i * indexMultiplier + moment2IndexOffset;

                // Adam复制momentum2输出
                DataCopy(this->optimizerState_.GetMomentum2OutGT()[theArgs.thisOutOffset], outLt[thisMoment2Index],
                         theArgs.embedDim);
            }
        }

        this->queOut_.FreeTensor(outLt);
    }

    // 实现CRTP接口
    __aicore__ inline void ComputeImpl(Args args)
    {
        Init(args);

        this->ClearGT(this->updateMaskGT_, this->updateMaskGT_.GetSize());
        ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();

        ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
    }

    // 更新权重调度器，直接使用FillUpdateArgs方法
    __aicore__ inline void UpdateWeightsSchedulerImpl(Args args)
    {
        InitOptimizer(args);
        TillingOptimizer();

        UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
        int64_t remain = this->workloadSharder_.length;
        while (remain > 0) {
            int64_t cnt = FillUpdateArgs(updateArgs, remain);
            DataCopyIn(updateArgs, cnt);
            ComputeOptimizer(updateArgs, cnt);
            DataCopyOut(updateArgs, cnt);
        }
    }

    TQue<TPosition::VECOUT, 1> queFlagOut;
    GlobalTensor<int8_t> updateMaskGT_; // 标记需更新的条目（NEED_UPDATE）
};

} // namespace BackwardCodegenUnweightedExact
#endif