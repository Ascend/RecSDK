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

#ifndef ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H
#define ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

namespace AsynchronousCompleteCumsum {

// 常量定义
constexpr int USE_QUEUE_NUM = 2;
constexpr int32_t BLOCK_SIZE = 256;                  // 每块元素数量
constexpr int32_t CACHE_LINE_SIZE = 64;              // Cache Line大小
constexpr int32_t DATA_ALIGN_BYTES = 32;             // 数据对齐字节数
constexpr int32_t RESERVER_UB_SIZE = 20 * 1024;      // UB保留空间
constexpr int32_t K_THRESHOLD = 48;                  // 中等规模阈值

// 参数结构体
struct Args {
    GM_ADDR input;
    GM_ADDR output;
    GM_ADDR workspace;
    GM_ADDR tiling;
};


template<typename T>
class AsynchronousCompleteCumsumKernel {
public:
    __aicore__ inline AsynchronousCompleteCumsumKernel(Args args)
    {
        // 获取tiling数据
        GET_TILING_DATA(tilingData, args.tiling);

        inputLength = tilingData.totalLength;
        totalBlocks = tilingData.totalBlocks;
        blocksPerCore = tilingData.blocksPerCore;
        remainderBlocks = tilingData.remainderBlocks;

        static_assert(sizeof(T) == sizeof(int32_t) || sizeof(T) == sizeof(int64_t), "T must be 4 or 8 bytes");
        cache_align = CACHE_LINE_SIZE / static_cast<int32_t>(sizeof(T));

        // 根据负载均衡策略计算每个core的工作分配
        int64_t coreId = GetBlockIdx();
        if (coreId < remainderBlocks) {
            myBlocksCount = blocksPerCore + 1;
            myStartBlock = coreId * myBlocksCount;
        } else {
            myBlocksCount = blocksPerCore;
            myStartBlock = remainderBlocks * (blocksPerCore + 1) +
                           (coreId - remainderBlocks) * blocksPerCore;
        }

        // 初始化gm内存
        inputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.input), (inputLength));
        outputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.output), (inputLength + 1));

        auto user_workspace = GetUserWorkspace(args.workspace);
        __gm__ T *addr = reinterpret_cast<__gm__ T *>(user_workspace);
        sharedMem.SetGlobalBuffer(addr, totalBlocks * cache_align);

        // 初始化UB资源
        pipe.InitBuffer(inputQueue, USE_QUEUE_NUM, BLOCK_SIZE * sizeof(T));
        pipe.InitBuffer(outputQueue, USE_QUEUE_NUM, (BLOCK_SIZE) * sizeof(T));
    }

    __aicore__ inline void Compute()
    {
        if (totalBlocks == 1) {
            ComputeBlockPrefixSums();
        } else {
            ComputeBlockPrefixSums();
            SyncAll();
            CombineResults();
        }
    }

private:
    // 数据拷贝函数
    __aicore__ inline void CopyGm2Local(const LocalTensor<T>& lt,
                                        const GlobalTensor<T>& gt, int32_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        DataCopy(lt, gt, alignLen / sizeof(T));
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
        }
    }

    __aicore__ inline void CopyLocal2Gm(const GlobalTensor<T>& gt,
                                        const LocalTensor<T>& lt, int32_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        DataCopy(gt, lt, alignLen / sizeof(T));
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
        }
    }

    // 第一阶段：只计算各块的部分和
    __aicore__ inline void ComputeBlockPrefixSums()
    {
        for (int64_t i = 0; i < myBlocksCount; i++) {
            int64_t blockIdx = myStartBlock + i;

            int64_t blockStart = blockIdx * BLOCK_SIZE;
            int64_t blockEnd = (blockStart + BLOCK_SIZE < inputLength) ? (blockStart + BLOCK_SIZE) : inputLength;
            int64_t actualSize = blockEnd - blockStart;

            T blockSum = static_cast<T>(0);
            for (int64_t j = 0; j < actualSize; ++j) {
                outputGT(blockStart + j) = blockSum;
                blockSum += inputGT(blockStart + j);
            }

            if (blockIdx == totalBlocks - 1) {
                outputGT(blockStart + actualSize) = blockSum;
            }

            sharedMem(blockIdx * cache_align) = blockSum;
            AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(sharedMem[blockIdx * cache_align]);
        }
    }

    // 第二阶段：计算完整前缀和，写入最终结果
    __aicore__ inline void CombineResults()
    {
        // 计算全局前缀和
        T prefixOffset = static_cast<T>(0);
        for (int64_t blockIdx = 0; blockIdx < myStartBlock; ++blockIdx) {
            T blockSum = sharedMem(blockIdx * cache_align);
            prefixOffset += blockSum;
        }

        if (myStartBlock == 0 && myBlocksCount > 1) {
            prefixOffset += sharedMem(0);
        }

        if constexpr (std::is_same_v<T, int64_t>) {
            CombineResultsInt64(prefixOffset);
        } else {
            CombineResultsInt32(prefixOffset);
        }
    }

    __aicore__ inline void CombineResultsInt64(T prefixOffset)
    {
        for (int64_t i = 0; i < myBlocksCount; i++) {
            int64_t blockIdx = myStartBlock + i;

            if (blockIdx == 0) {
                continue;
            }

            int64_t blockStart = blockIdx * BLOCK_SIZE;
            int64_t blockEnd = (blockStart + BLOCK_SIZE < inputLength) ? (blockStart + BLOCK_SIZE) : inputLength;
            int64_t actualSize = blockEnd - blockStart;

            GlobalTensor<T> outputSlice = outputGT[blockStart];
            for (int64_t j = 0; j < actualSize; ++j) {
                outputSlice(j) += prefixOffset;
            }

            if (blockIdx == totalBlocks - 1) {
                outputGT(blockStart + actualSize) += prefixOffset;
            }

            prefixOffset += sharedMem(blockIdx * cache_align);
        }
    }

    __aicore__ inline void CombineResultsInt32(T prefixOffset)
    {
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(sharedMem);

        for (int64_t i = 0; i < myBlocksCount; ++i) {
            int64_t blockIdx = myStartBlock + i;

            if (blockIdx == 0) {
                continue;
            }

            int64_t blockStart = blockIdx * BLOCK_SIZE;
            int64_t blockEnd = (blockStart + BLOCK_SIZE < inputLength) ? (blockStart + BLOCK_SIZE) : inputLength;
            int64_t actualSize = blockEnd - blockStart;
            int64_t leftSize = 0;

            if (blockIdx == totalBlocks - 1) {
                int64_t totalSize = actualSize;
                int64_t alignBytes = totalSize * sizeof(T) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;
                actualSize = alignBytes / sizeof(T);  // 对齐部分的元素个数

                int64_t unalignBytes = totalSize * sizeof(T) - alignBytes;
                leftSize = unalignBytes / sizeof(T);  // 非对齐部分的元素个数
            }

            GlobalTensor<T> outputSlice = outputGT[blockStart];
            LocalTensor<T> localIn = inputQueue.AllocTensor<T>();
            CopyGm2Local(localIn, outputSlice, actualSize);

            inputQueue.EnQue(localIn);
            LocalTensor<T> localInCopy = inputQueue.DeQue<T>();
            LocalTensor<T> finalResults = outputQueue.AllocTensor<T>();

            Adds(finalResults, localInCopy, prefixOffset, actualSize);

            outputQueue.EnQue(finalResults);
            LocalTensor<T> finalResultsCopy = outputQueue.DeQue<T>();
            CopyLocal2Gm(outputSlice, finalResultsCopy, actualSize);

            outputQueue.FreeTensor(finalResultsCopy);
            inputQueue.FreeTensor(localInCopy);

            if (blockIdx == totalBlocks - 1) {
                for (int64_t j = 0; j <= leftSize; ++j) {
                    outputGT(blockStart + actualSize + j) += prefixOffset;
                }
            }

            prefixOffset += sharedMem(blockIdx * cache_align);
        }
    }

private:
    // GM_ADDR
    GM_ADDR input;
    GM_ADDR output;
    GM_ADDR workspace;
    GM_ADDR tiling;

    // Tiling参数
    int64_t inputLength;
    int64_t totalBlocks;
    int64_t blocksPerCore;
    int64_t remainderBlocks;
    int32_t cache_align;

    // 当前core信息
    int64_t myBlocksCount;
    int64_t myStartBlock;

    // Global Tensor
    GlobalTensor<T> inputGT;
    GlobalTensor<T> outputGT;
    GlobalTensor<T> sharedMem;

    // Queue
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
};

} // namespace AsynchronousCompleteCumsum

#endif // ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H