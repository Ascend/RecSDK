/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MASKSELECT_OP_KERNEL_H
#define MASKSELECT_OP_KERNEL_H

#include "simt_api/asc_simt.h"
#include "maskselect_simt.h"
#include "maskselect_common.h"

struct MaskSelectArgs {
    __gm__ uint8_t* flags;
    __gm__ void* inputs;
    __gm__ void* outputs;
    __gm__ int64_t* numSelected;
    __gm__ int64_t* workspace;
    int64_t numTotal;
    int32_t isSmall;
    int32_t isFullCore;
    int32_t totalBlocks;
    int32_t blocksPerCore;
    int32_t remainderBlocks;
    int32_t elementsPerBlock;
};

namespace MaskSelect {

using namespace MaskSelectSimt;

constexpr int BUFFER_NUM = MASKSELECT_BUFFER_NUM;

template <typename KeyT, bool SelectIndex>
class MaskSelectKernel {
public:
    __aicore__ inline MaskSelectKernel(MaskSelectArgs& args)
    {
        InitTilingParams(args);
        InitGmParams(args);
        InitUbParams();
    }

    __aicore__ inline void Compute()
    {
        int32_t coreIdx = GetBlockIdx();
        if (coreIdx < remainderBlocks_) {
            blockCount_ = blocksPerCore_ + 1;
            blockStart_ = coreIdx * blockCount_;
        } else {
            blockCount_ = blocksPerCore_;
            blockStart_ = remainderBlocks_ * (blocksPerCore_ + 1) + (coreIdx - remainderBlocks_) * blocksPerCore_;
        }

        if (totalLength_ <= 0) {
            if (coreIdx == 0) {
                numSelectedGT.SetValue(0, static_cast<int64_t>(0));
                AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                  AscendC::DcciDst::CACHELINE_OUT>(numSelectedGT[0]);
            }
            return;
        }

        if (isFullCore_) {
            ProcessPrefixMultiCycles();
        } else {
            ProcessPrefixOneCycle();
        }

        SyncAll();
        AscendC::PipeBarrier<PIPE_ALL>();

        ProcessScatter();
        SyncAll();

        if (coreIdx == 0) {
            WriteNumSelected();
        }
        SyncAll();
    }

private:
    __aicore__ inline void InitTilingParams(const MaskSelectArgs& args)
    {
        totalLength_ = args.numTotal;
        totalBlocks_ = args.totalBlocks;
        blocksPerCore_ = args.blocksPerCore;
        remainderBlocks_ = args.remainderBlocks;
        elementsPerBlock_ = args.elementsPerBlock;
        isSmall_ = args.isSmall > 0;
        isFullCore_ = args.isFullCore > 0;
    }

    __aicore__ inline void InitGmParams(const MaskSelectArgs& args)
    {
        flagsGT.SetGlobalBuffer(args.flags, totalLength_);
        flagsGm = args.flags;
        if constexpr (!SelectIndex) {
            keysGT.SetGlobalBuffer(reinterpret_cast<__gm__ KeyT*>(args.inputs), totalLength_);
        }
        outputsGT.SetGlobalBuffer(reinterpret_cast<__gm__ KeyT*>(args.outputs), totalLength_);
        outputsGmPtr = reinterpret_cast<__gm__ KeyT*>(args.outputs);
        numSelectedGT.SetGlobalBuffer(args.numSelected, 1);

        prefixGm = args.workspace;
        blockSumsGm = prefixGm + totalLength_;
    }

    __aicore__ inline void InitUbParams()
    {
        pipe.InitBuffer(flagsQueue, BUFFER_NUM, elementsPerBlock_ * sizeof(uint8_t));
        pipe.InitBuffer(prefixQueue, BUFFER_NUM, elementsPerBlock_ * sizeof(int64_t));
        if constexpr (!SelectIndex) {
            pipe.InitBuffer(keysQueue, BUFFER_NUM, elementsPerBlock_ * sizeof(KeyT));
        }
        pipe.InitBuffer(sharedBuf, MASKSELECT_MAX_WARPS * sizeof(int64_t));
        pipe.InitBuffer(reduceTmpBuf, elementsPerBlock_ * sizeof(int64_t));
        pipe.InitBuffer(reduceDstBuf, MASKSELECT_DATA_ALIGN_BYTES);
    }

    __aicore__ inline int64_t ComputePrefixOffset(int32_t blockIdx, LocalTensor<int64_t>& scratchLt)
    {
        if (blockIdx == 0) {
            return 0;
        }

        MaskSelectInvalidateInt64GmCache(blockSumsGm, blockIdx);

        LocalTensor<uint8_t> reduceTmpLt = reduceTmpBuf.Get<uint8_t>();
        LocalTensor<int64_t> reduceDstLt = reduceDstBuf.Get<int64_t>();
        GlobalTensor<int64_t> blockSumGT;
        blockSumGT.SetGlobalBuffer(blockSumsGm, totalBlocks_);

        int64_t prefix = 0;
        int32_t processed = 0;
        while (processed < blockIdx) {
            int32_t chunk = blockIdx - processed;
            chunk = chunk > elementsPerBlock_ ? elementsPerBlock_ : chunk;

            CpGm2Local(scratchLt, blockSumGT[processed], chunk);
            prefixQueue.EnQue(scratchLt);
            scratchLt = prefixQueue.DeQue<int64_t>();
            uint32_t shape[2] = {1, static_cast<uint32_t>(chunk)};
            ReduceSum<int64_t, AscendC::Pattern::Reduce::AR>(reduceDstLt, scratchLt, reduceTmpLt, shape, false);
            prefix += reduceDstLt(0);
            processed += chunk;
        }
        return prefix;
    }

    __aicore__ inline void ProcessPrefixOneCycle()
    {
        int32_t blockBase = blockStart_ * elementsPerBlock_;
        int32_t remain = static_cast<int32_t>(totalLength_) - blockBase;
        int32_t elementsThisBlock = remain < elementsPerBlock_ ? remain : elementsPerBlock_;

        LocalTensor<uint8_t> flagsLt = flagsQueue.AllocTensor<uint8_t>();
        CpGm2Local(flagsLt, flagsGT[blockBase], elementsThisBlock);
        flagsQueue.EnQue(flagsLt);
        flagsLt = flagsQueue.DeQue<uint8_t>();

        LocalTensor<int64_t> prefixLt = prefixQueue.AllocTensor<int64_t>();
        LocalTensor<int64_t> sharedLt = sharedBuf.Get<int64_t>();
        __ubuf__ int64_t* sharedUb = (__ubuf__ int64_t*)sharedLt.GetPhyAddr();
        __ubuf__ uint8_t* flagsLocal = (__ubuf__ uint8_t*)flagsLt.GetPhyAddr();
        __ubuf__ int64_t* prefixLocal = (__ubuf__ int64_t*)prefixLt.GetPhyAddr();

        if (isSmall_) {
            asc_vf_call<MaskSelectSimt::SmallFlagPrefixCompute<int64_t>>(dim3{MASKSELECT_MAX_THREADS, 1, 1}, flagsLocal,
                                                                         prefixLocal, blockSumsGm, sharedUb,
                                                                         elementsThisBlock, blockStart_);
        } else {
            asc_vf_call<MaskSelectSimt::LargeFlagPrefixCompute<int64_t>>(dim3{MASKSELECT_MAX_THREADS, 1, 1}, flagsLocal,
                                                                         prefixLocal, blockSumsGm, sharedUb,
                                                                         elementsThisBlock, blockStart_);
        }
        if (totalBlocks_ > 1) {
            MaskSelectFlushBlockSumGm(blockSumsGm, blockStart_);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if (totalBlocks_ == 1) {
            flagsQueue.FreeTensor(flagsLt);
            prefixQueue.EnQue(prefixLt);
            prefixLt = prefixQueue.DeQue<int64_t>();
            CpLocal2GmInt64AfterCompute(prefixGm + blockBase, prefixLt, elementsThisBlock);
            prefixQueue.FreeTensor(prefixLt);
        } else {
            SyncAll();
            flagsQueue.FreeTensor(flagsLt);
            int64_t prefixOffset = 0;
            {
                LocalTensor<int64_t> scratchLt = prefixQueue.AllocTensor<int64_t>();
                prefixOffset = ComputePrefixOffset(blockStart_, scratchLt);
                prefixQueue.FreeTensor(scratchLt);
            }
            Adds(prefixLt, prefixLt, prefixOffset, elementsThisBlock);
            prefixQueue.EnQue(prefixLt);
            prefixLt = prefixQueue.DeQue<int64_t>();
            CpLocal2GmInt64AfterCompute(prefixGm + blockBase, prefixLt, elementsThisBlock);
            prefixQueue.FreeTensor(prefixLt);
        }
    }

    __aicore__ inline void ProcessPrefixMultiCycles()
    {
        GlobalTensor<int64_t> prefixGT;
        prefixGT.SetGlobalBuffer(prefixGm, totalLength_);
        LocalTensor<int64_t> sharedLt = sharedBuf.Get<int64_t>();
        __ubuf__ int64_t* sharedUb = (__ubuf__ int64_t*)sharedLt.GetPhyAddr();

        for (int32_t blockIdx = blockStart_; blockIdx < blockStart_ + blockCount_; ++blockIdx) {
            int32_t blockBase = blockIdx * elementsPerBlock_;
            int32_t remain = static_cast<int32_t>(totalLength_) - blockBase;
            int32_t elementsThisBlock = remain < elementsPerBlock_ ? remain : elementsPerBlock_;

            LocalTensor<uint8_t> flagsLt = flagsQueue.AllocTensor<uint8_t>();
            CpGm2Local(flagsLt, flagsGT[blockBase], elementsThisBlock);
            flagsQueue.EnQue(flagsLt);
            flagsLt = flagsQueue.DeQue<uint8_t>();

            LocalTensor<int64_t> prefixLt = prefixQueue.AllocTensor<int64_t>();
            __ubuf__ uint8_t* flagsLocal = (__ubuf__ uint8_t*)flagsLt.GetPhyAddr();
            __ubuf__ int64_t* prefixLocal = (__ubuf__ int64_t*)prefixLt.GetPhyAddr();

            if (isSmall_) {
                asc_vf_call<MaskSelectSimt::SmallFlagPrefixCompute<int64_t>>(dim3{MASKSELECT_MAX_THREADS, 1, 1},
                                                                             flagsLocal, prefixLocal, blockSumsGm,
                                                                             sharedUb, elementsThisBlock, blockIdx);
            } else {
                asc_vf_call<MaskSelectSimt::LargeFlagPrefixCompute<int64_t>>(dim3{MASKSELECT_MAX_THREADS, 1, 1},
                                                                             flagsLocal, prefixLocal, blockSumsGm,
                                                                             sharedUb, elementsThisBlock, blockIdx);
            }
            if (totalBlocks_ > 1) {
                MaskSelectFlushBlockSumGm(blockSumsGm, blockIdx);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            flagsQueue.FreeTensor(flagsLt);
            prefixQueue.EnQue(prefixLt);
            prefixLt = prefixQueue.DeQue<int64_t>();
            CpLocal2GmInt64AfterCompute(prefixGm + blockBase, prefixLt, elementsThisBlock);
            prefixQueue.FreeTensor(prefixLt);
        }

        SyncAll();

        int64_t prefixOffset = 0;
        {
            LocalTensor<int64_t> scratchLt = prefixQueue.AllocTensor<int64_t>();
            prefixOffset = ComputePrefixOffset(blockStart_, scratchLt);
            prefixQueue.FreeTensor(scratchLt);
        }

        LocalTensor<int64_t> blockSumScratchLt = sharedBuf.Get<int64_t>();
        for (int32_t blockIdx = blockStart_; blockIdx < blockStart_ + blockCount_; ++blockIdx) {
            int32_t blockBase = blockIdx * elementsPerBlock_;
            int32_t remain = static_cast<int32_t>(totalLength_) - blockBase;
            int32_t elementsThisBlock = remain < elementsPerBlock_ ? remain : elementsPerBlock_;

            LocalTensor<int64_t> prefixLt = prefixQueue.AllocTensor<int64_t>();
            CpGm2Local(prefixLt, prefixGT[blockBase], elementsThisBlock);
            prefixQueue.EnQue(prefixLt);
            prefixLt = prefixQueue.DeQue<int64_t>();

            Adds(prefixLt, prefixLt, prefixOffset, elementsThisBlock);
            prefixQueue.EnQue(prefixLt);
            prefixLt = prefixQueue.DeQue<int64_t>();
            CpLocal2GmInt64AfterCompute(prefixGm + blockBase, prefixLt, elementsThisBlock);
            prefixQueue.FreeTensor(prefixLt);

            prefixOffset += ReadBlockSumGm(blockSumsGm, blockIdx, blockSumScratchLt);
        }
    }

    __aicore__ inline void ProcessScatterBlock(int32_t blockIdx)
    {
        int32_t blockBase = blockIdx * elementsPerBlock_;
        int32_t remain = static_cast<int32_t>(totalLength_) - blockBase;
        int32_t elementsThisBlock = remain < elementsPerBlock_ ? remain : elementsPerBlock_;
        if (elementsThisBlock <= 0) {
            return;
        }

        LocalTensor<uint8_t> flagsLt = flagsQueue.AllocTensor<uint8_t>();
        CpGm2Local(flagsLt, flagsGT[blockBase], elementsThisBlock);
        flagsQueue.EnQue(flagsLt);
        flagsLt = flagsQueue.DeQue<uint8_t>();

        LocalTensor<int64_t> prefixLt = prefixQueue.AllocTensor<int64_t>();
        GlobalTensor<int64_t> prefixGT;
        prefixGT.SetGlobalBuffer(prefixGm, totalLength_);
        CpGm2Local(prefixLt, prefixGT[blockBase], elementsThisBlock);
        prefixQueue.EnQue(prefixLt);
        prefixLt = prefixQueue.DeQue<int64_t>();

        __ubuf__ uint8_t* flagsLocal = (__ubuf__ uint8_t*)flagsLt.GetPhyAddr();
        __ubuf__ int64_t* prefixLocal = (__ubuf__ int64_t*)prefixLt.GetPhyAddr();

        if constexpr (!SelectIndex) {
            LocalTensor<KeyT> keysLt = keysQueue.AllocTensor<KeyT>();
            CpGm2Local(keysLt, keysGT[blockBase], elementsThisBlock);
            keysQueue.EnQue(keysLt);
            keysLt = keysQueue.DeQue<KeyT>();
            __ubuf__ KeyT* keysLocal = (__ubuf__ KeyT*)keysLt.GetPhyAddr();
            if (isSmall_) {
                asc_vf_call<MaskSelectSimt::SmallScatterCompute<KeyT, int64_t, SelectIndex>>(
                    dim3{MASKSELECT_MAX_THREADS, 1, 1}, flagsLocal, keysLocal, prefixLocal, outputsGmPtr,
                    elementsThisBlock, blockBase);
            } else {
                asc_vf_call<MaskSelectSimt::LargeScatterCompute<KeyT, int64_t, SelectIndex>>(
                    dim3{MASKSELECT_MAX_THREADS, 1, 1}, flagsLocal, keysLocal, prefixLocal, outputsGmPtr,
                    elementsThisBlock, blockBase);
            }
            keysQueue.FreeTensor(keysLt);
        } else {
            if (isSmall_) {
                asc_vf_call<MaskSelectSimt::SmallScatterCompute<KeyT, int64_t, SelectIndex>>(
                    dim3{MASKSELECT_MAX_THREADS, 1, 1}, flagsLocal, (__ubuf__ KeyT*)nullptr, prefixLocal, outputsGmPtr,
                    elementsThisBlock, blockBase);
            } else {
                asc_vf_call<MaskSelectSimt::LargeScatterCompute<KeyT, int64_t, SelectIndex>>(
                    dim3{MASKSELECT_MAX_THREADS, 1, 1}, flagsLocal, (__ubuf__ KeyT*)nullptr, prefixLocal, outputsGmPtr,
                    elementsThisBlock, blockBase);
            }
        }

        flagsQueue.FreeTensor(flagsLt);
        prefixQueue.FreeTensor(prefixLt);
    }

    __aicore__ inline void ProcessScatter()
    {
        for (int32_t blockIdx = blockStart_; blockIdx < blockStart_ + blockCount_; ++blockIdx) {
            ProcessScatterBlock(blockIdx);
        }
    }

    __aicore__ inline void WriteNumSelected()
    {
        LocalTensor<int64_t> scratchLt = sharedBuf.Get<int64_t>();

        GlobalTensor<int64_t> prefixTailGt;
        prefixTailGt.SetGlobalBuffer(prefixGm + totalLength_ - 1, 1);
        MaskSelectInvalidateInt64GmCache(prefixGm + totalLength_ - 1, 1);
        CpGm2Local(scratchLt, prefixTailGt, 1);
        ops_utils::SyncMte2V();
        int64_t count = scratchLt.GetValue(0);

        GlobalTensor<uint8_t> flagTailGt;
        flagTailGt.SetGlobalBuffer(flagsGm + totalLength_ - 1, 1);
        LocalTensor<uint8_t> flagScratchLt = reduceTmpBuf.Get<uint8_t>();
        CpGm2Local(flagScratchLt, flagTailGt, 1);
        ops_utils::SyncMte2V();
        if (flagScratchLt.GetValue(0) != 0) {
            count += 1;
        }

        numSelectedGT.SetValue(0, count);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(numSelectedGT[0]);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> flagsQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> prefixQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> keysQueue;
    TBuf<TPosition::VECCALC> sharedBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;

    GlobalTensor<uint8_t> flagsGT;
    GlobalTensor<KeyT> keysGT;
    GlobalTensor<KeyT> outputsGT;
    GlobalTensor<int64_t> numSelectedGT;

    __gm__ uint8_t* flagsGm;
    __gm__ int64_t* prefixGm;
    __gm__ int64_t* blockSumsGm;
    __gm__ KeyT* outputsGmPtr;

    int64_t totalLength_;
    int32_t totalBlocks_;
    int32_t blocksPerCore_;
    int32_t remainderBlocks_;
    int32_t elementsPerBlock_;
    bool isSmall_;
    bool isFullCore_;
    int32_t blockCount_;
    int32_t blockStart_;
};

}  // namespace MaskSelect

#define KEY_TYPE_DISPATCH(isUInt64, KEY_TYPE, ...) \
    do {                                           \
        if (isUInt64) {                            \
            using KEY_TYPE = uint64_t;             \
            __VA_ARGS__;                           \
        } else {                                   \
            using KEY_TYPE = int64_t;              \
            __VA_ARGS__;                           \
        }                                          \
    } while (0)

namespace dyn_emb {

__global__ __vector__ void maskselect_op(__gm__ bool* flags, __gm__ int64_t* inputs, __gm__ int64_t* outputs,
                                         __gm__ int64_t* numSelected, __gm__ int64_t* workspace, int64_t numTotal,
                                         int32_t isUInt64, int32_t isSmall, int32_t isFullCore, int32_t totalBlocks,
                                         int32_t blocksPerCore, int32_t remainderBlocks, int32_t elementsPerBlock)
{
    KEY_TYPE_DISPATCH(isUInt64 > 0, KeyType, {
        MaskSelectArgs args{reinterpret_cast<__gm__ uint8_t*>(flags),
                            static_cast<__gm__ void*>(reinterpret_cast<__gm__ KeyType*>(inputs)),
                            static_cast<__gm__ void*>(reinterpret_cast<__gm__ KeyType*>(outputs)),
                            numSelected,
                            workspace,
                            numTotal,
                            isSmall,
                            isFullCore,
                            totalBlocks,
                            blocksPerCore,
                            remainderBlocks,
                            elementsPerBlock};
        MaskSelect::MaskSelectKernel<KeyType, false> kernel(args);
        kernel.Compute();
    });
}

__global__ __vector__ void maskselect_index_op(__gm__ bool* flags, __gm__ int64_t* outputIndices,
                                               __gm__ int64_t* numSelected, __gm__ int64_t* workspace, int64_t numTotal,
                                               int32_t isUInt64, int32_t isSmall, int32_t isFullCore,
                                               int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks,
                                               int32_t elementsPerBlock)
{
    KEY_TYPE_DISPATCH(isUInt64 > 0, KeyType, {
        MaskSelectArgs args{reinterpret_cast<__gm__ uint8_t*>(flags),
                            static_cast<__gm__ void*>(nullptr),
                            static_cast<__gm__ void*>(reinterpret_cast<__gm__ KeyType*>(outputIndices)),
                            numSelected,
                            workspace,
                            numTotal,
                            isSmall,
                            isFullCore,
                            totalBlocks,
                            blocksPerCore,
                            remainderBlocks,
                            elementsPerBlock};
        MaskSelect::MaskSelectKernel<KeyType, true> kernel(args);
        kernel.Compute();
    });
}

}  // namespace dyn_emb

#endif  // MASKSELECT_OP_KERNEL_H
