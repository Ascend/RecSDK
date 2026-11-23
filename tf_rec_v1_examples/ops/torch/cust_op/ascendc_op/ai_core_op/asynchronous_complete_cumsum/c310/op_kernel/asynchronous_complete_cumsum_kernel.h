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

#ifndef ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H
#define ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H

#include "simt_kernel.h"
#include "kernel_common_utils.h"

struct Args {
    GM_ADDR x;
    GM_ADDR y;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

namespace AsynchronousCompleteCumsum {

constexpr int BUFFER_NUM = 2;

template <typename T>
class AsynchronousCompleteCumsumKernel {
public:
    __aicore__ inline AsynchronousCompleteCumsumKernel(Args &args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        InitTilingParams(tilingData);
        InitGmParams(args);
        InitUbParams();
    }

    __aicore__ inline void Compute()
    {
        int32_t coreIdx = GetBlockIdx();
        if (coreIdx < remainderBlocks) {
            blockCount = blocksPerCore + 1;
            blockStart = coreIdx * blockCount;
        } else {
            blockCount = blocksPerCore;
            blockStart = remainderBlocks * (blocksPerCore + 1) + (coreIdx - remainderBlocks) * blocksPerCore;
        }

        if (coreIdx == 0) {
            outputGT.SetValue(0, static_cast<T>(0));
            AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(outputGT[0]);
        }

        if (isFullCore) {
            ProcessMultiCycles();
        } else {
            ProcessOneCycle();
        }
    }

private:
    __aicore__ inline void InitTilingParams(const AsynchronousCompleteCumsumTilingData &tilingData)
    {
        totalLength = tilingData.totalLength;
        totalBlocks = tilingData.totalBlocks;
        blocksPerCore = tilingData.blocksPerCore;
        remainderBlocks = tilingData.remainderBlocks;
        elementsPerBlock = tilingData.elementsPerBlock;
        isSmall = tilingData.isSmall;
        isFullCore = tilingData.isFullCore;
    }

    __aicore__ inline void InitGmParams(const Args &args)
    {
        inputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(args.x), totalLength);
        outputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(args.y), totalLength + 1);

        auto user_workspace = GetUserWorkspace(args.workspace);
        sharedMem = reinterpret_cast<__gm__ T*>(user_workspace);
        blockSumGT.SetGlobalBuffer(sharedMem, totalBlocks);
    }

    __aicore__ inline void InitUbParams()
    {
        pipe.InitBuffer(inputQueue, BUFFER_NUM, elementsPerBlock * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, elementsPerBlock * sizeof(T));
        pipe.InitBuffer(sharedBuf, MAX_WARPS * sizeof(T));
        pipe.InitBuffer(reduceTmpBuf, elementsPerBlock * sizeof(T));
        pipe.InitBuffer(reduceDstBuf, DATA_ALIGN_BYTES);
    }

    __aicore__ inline T ComputePrefixOffset(int64_t blockIdx, LocalTensor<T> &scratchLt)
    {
        if (blockIdx == 0) {
            return static_cast<T>(0);
        }

        LocalTensor<uint8_t> reduceTmpLt = reduceTmpBuf.Get<uint8_t>();
        LocalTensor<T> reduceDstLt = reduceDstBuf.Get<T>();

        T prefix = static_cast<T>(0);
        int64_t processed = 0;
        while (processed < blockIdx) {
            int32_t chunk = static_cast<int32_t>(blockIdx - processed);
            chunk = chunk > elementsPerBlock ? elementsPerBlock : chunk;

            CpGm2Local(scratchLt, blockSumGT[processed], chunk);
            inputQueue.EnQue(scratchLt);
            scratchLt = inputQueue.DeQue<T>();
            uint32_t shape[2] = {1, static_cast<uint32_t>(chunk)};
            ReduceSum<T, AscendC::Pattern::Reduce::AR>(reduceDstLt, scratchLt, reduceTmpLt, shape, false);
            prefix += reduceDstLt(0);
            processed += chunk;
        }
        return prefix;
    }

    __aicore__ inline void SimtProcess(__local_mem__ T* input, __local_mem__ T* output, __ubuf__ T* sharedUb,
                                       int32_t elementsThisBlock, int64_t blockIdx)
    {
        if (isSmall) {
            asc_vf_call<CumsumSimt::SmallDataCompute<T>>(
                dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                input,
                output,
                sharedMem,
                sharedUb,
                elementsThisBlock,
                blockIdx);
        } else {
            asc_vf_call<CumsumSimt::LargeDataCompute<T>>(
                dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                input,
                output,
                sharedMem,
                sharedUb,
                elementsThisBlock,
                blockIdx);
        }
    }

    __aicore__ inline void ProcessOneCycle()
    {
        // 每个AI Core只处理一个logical block
        int64_t blockBase = blockStart * elementsPerBlock;
        int64_t remain = totalLength - blockBase;
        int32_t elementsThisBlock = static_cast<int32_t>(remain < elementsPerBlock ? remain : elementsPerBlock);

        LocalTensor<T> sharedLt = sharedBuf.Get<T>();
        __ubuf__ T* sharedUb =  reinterpret_cast<__ubuf__ T*>(sharedLt.GetPhyAddr());

        LocalTensor<T> inputLt = inputQueue.AllocTensor<T>();
        CpGm2Local(inputLt, inputGT[blockBase], elementsThisBlock);
        inputQueue.EnQue(inputLt);
        inputLt = inputQueue.DeQue<T>();
        __local_mem__ T* input = reinterpret_cast<__local_mem__ T*>(inputLt.GetPhyAddr());

        LocalTensor<T> outputLt = outputQueue.AllocTensor<T>();
        __local_mem__ T* output = reinterpret_cast<__local_mem__ T*>(outputLt.GetPhyAddr());

        SimtProcess(input, output, sharedUb, elementsThisBlock, blockStart);

        if (totalBlocks == 1) {
            inputQueue.FreeTensor(inputLt);
            outputQueue.EnQue(outputLt);
            outputLt = outputQueue.DeQue<T>();
            CpLocal2Gm(outputGT[blockBase + 1], outputLt, elementsThisBlock);
            outputQueue.FreeTensor(outputLt);
        } else {
            SyncAll();

            // 计算当前线程块之前所有线程块的总和prefixOffset
            T prefixOffset = ComputePrefixOffset(blockStart, inputLt);
            inputQueue.FreeTensor(inputLt);

            Adds(outputLt, outputLt, prefixOffset, elementsThisBlock);
            outputQueue.EnQue(outputLt);
            outputLt = outputQueue.DeQue<T>();
            CpLocal2Gm(outputGT[blockBase + 1], outputLt, elementsThisBlock);
            outputQueue.FreeTensor(outputLt);
        }
    }

    __aicore__ inline void ProcessMultiCycles()
    {
        LocalTensor<T> sharedLt = sharedBuf.Get<T>();
        __ubuf__ T* sharedUb =  reinterpret_cast<__ubuf__ T*>(sharedLt.GetPhyAddr());

        // 第一阶段：块内前缀和写回GM
        for (int64_t blockIdx = blockStart; blockIdx < blockStart + blockCount; ++blockIdx) {
            int64_t blockBase = blockIdx * elementsPerBlock;
            int64_t remain = totalLength - blockBase;
            int32_t elementsThisBlock = static_cast<int32_t>(remain < elementsPerBlock ? remain : elementsPerBlock);

            LocalTensor<T> inputLt = inputQueue.AllocTensor<T>();
            CpGm2Local(inputLt, inputGT[blockBase], elementsThisBlock);
            inputQueue.EnQue(inputLt);
            inputLt = inputQueue.DeQue<T>();
            __local_mem__ T* input = reinterpret_cast<__local_mem__ T*>(inputLt.GetPhyAddr());

            LocalTensor<T> outputLt = outputQueue.AllocTensor<T>();
            __local_mem__ T* output = reinterpret_cast<__local_mem__ T*>(outputLt.GetPhyAddr());

            SimtProcess(input, output, sharedUb, elementsThisBlock, blockIdx);

            inputQueue.FreeTensor(inputLt);
            outputQueue.EnQue(outputLt);
            outputLt = outputQueue.DeQue<T>();
            CpLocal2Gm(outputGT[blockBase + 1], outputLt, elementsThisBlock);
            outputQueue.FreeTensor(outputLt);
        }

        SyncAll();

        // 计算该核心首个block的跨block偏移
        LocalTensor<T> inputLt = inputQueue.AllocTensor<T>();
        T prefixOffset = ComputePrefixOffset(blockStart, inputLt);
        inputQueue.FreeTensor(inputLt);

        for (int64_t blockIdx = blockStart; blockIdx < blockStart + blockCount; ++blockIdx) {
            int64_t blockBase = blockIdx * elementsPerBlock;
            int64_t remain = totalLength - blockBase;
            int32_t elementsThisBlock = static_cast<int32_t>(remain < elementsPerBlock ? remain : elementsPerBlock);

            LocalTensor<T> inputLt = inputQueue.AllocTensor<T>();
            CpGm2Local(inputLt, outputGT[blockBase + 1], elementsThisBlock);
            inputQueue.EnQue(inputLt);
            inputLt = inputQueue.DeQue<T>();

            LocalTensor<T> correctionLt = outputQueue.AllocTensor<T>();
            Adds(correctionLt, inputLt, prefixOffset, elementsThisBlock);
            outputQueue.EnQue(correctionLt);
            correctionLt = outputQueue.DeQue<T>();
            CpLocal2Gm(outputGT[blockBase + 1], correctionLt, elementsThisBlock);
            outputQueue.FreeTensor(correctionLt);
            inputQueue.FreeTensor(inputLt);

            prefixOffset += sharedMem[blockIdx];
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> sharedBuf;
    TBuf<TPosition::VECCALC> reduceTmpBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;

    GlobalTensor<T> inputGT;
    GlobalTensor<T> outputGT;
    GlobalTensor<T> blockSumGT;
    __gm__ T* sharedMem;

    int64_t totalLength;
    int64_t totalBlocks;
    int64_t blocksPerCore;
    int32_t remainderBlocks;
    int32_t elementsPerBlock;
    bool isSmall;
    bool isFullCore;
    int64_t blockCount;
    int64_t blockStart;
};

}  // namespace AsynchronousCompleteCumsum

#endif  // ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H
