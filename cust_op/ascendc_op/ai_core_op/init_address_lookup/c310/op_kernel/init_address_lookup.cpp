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

#include "kernel_operator.h"
#include "kernel_common_utils.h"
using namespace AscendC;

namespace InitAddressLookup {

constexpr int DATA_ALIGN_BYTES = 32;

template <typename T>
class KernelInitAddressLookup {
public:
    __aicore__ inline KernelInitAddressLookup() {}

    __aicore__ inline void Init(GM_ADDR bufferOffsets, GM_ADDR embSizes, GM_ADDR addressLookups,
                                GM_ADDR addressLookupsOut, GM_ADDR tiling)
    {
        // 获取Tiling参数
        GET_TILING_DATA(tilingData, tiling);
        numTables = tilingData.num_tables;
        totalRows = tilingData.total_rows;
        lutSize = tilingData.lut_size;
        coreNum = tilingData.core_num;
        if (lutSize <= 0) {
            lutSize = 4;
        }

        // 如果当前核无效，直接返回
        if (block_idx >= coreNum) {
            needProcess = false;
            return;
        }
        needProcess = true;

        // 初始化GlobalTensor
        bufferOffsetsGm.SetGlobalBuffer((__gm__ int64_t*)bufferOffsets, numTables + 1);
        embSizesGm.SetGlobalBuffer((__gm__ T*)embSizes, numTables);
        // addressLookups是输入（已分配的输出buffer），addressLookupsOut是输出（同一块内存）
        addressLookupsGm.SetGlobalBuffer((__gm__ T*)addressLookupsOut, totalRows);

        // 分配LUT缓冲 + 输出缓冲
        int64_t lutBufferBytes = lutSize * static_cast<int64_t>(sizeof(T));
        lutBufferBytes = (lutBufferBytes + 31) & ~31;
        pipe.InitBuffer(lutBuf, lutBufferBytes);
        pipe.InitBuffer(outQueue, 1, lutBufferBytes);
    }

    __aicore__ inline void Process()
    {
        if (!needProcess) {
            return;
        }
        ProcessMultiCoreSplit();
    }

    // 在 bufferOffsetsGm 上二分查找，找到 pos 所属的表索引
    __aicore__ inline int64_t FindTable(int64_t pos)
    {
        int64_t lo = 0;
        int64_t hi = numTables;
        while (lo < hi) {
            int64_t mid = (lo + hi + 1) / 2;
            if (static_cast<int64_t>(bufferOffsetsGm.GetValue(mid)) <= pos) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    // 全局行切分，将 totalRows 按输出位置均分给所有核
    __aicore__ inline void ProcessMultiCoreSplit()
    {
        // Kernel-side tiling：按输出位置均分给所有核，切分边界 128B 元素对齐
        constexpr int64_t ELEMS_PER_BLOCK = DATA_ALIGN_BYTES * 4 / static_cast<int64_t>(sizeof(T));
        int64_t bid = static_cast<int64_t>(block_idx);

        // 计算起始点
        const int64_t rawStart = (bid == 0) ? static_cast<int64_t>(0) : (totalRows * bid) / coreNum;
        const int64_t coreStart = (bid == 0) ? static_cast<int64_t>(0)
                                             : ((rawStart + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;

        // 计算结束点
        const int64_t rawEnd = (totalRows * (bid + 1)) / coreNum;
        const int64_t alignedEnd = ((rawEnd + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;
        const int64_t coreEnd = (bid == coreNum - 1) ? totalRows : (alignedEnd > totalRows ? totalRows : alignedEnd);

        if (coreStart >= coreEnd) {
            return;
        }

        // 初始化LUT: [0, 1, 2, ..., lutSize-1]
        LocalTensor<T> lutLT = lutBuf.Get<T>();
        CreateVecIndex(lutLT, static_cast<T>(0), static_cast<uint32_t>(lutSize));

        // 二分查找 coreStart 所属的表
        int64_t tableIdx = FindTable(coreStart);
        int64_t pos = coreStart;

        while (pos < coreEnd && tableIdx < numTables) {
            int64_t tableStart = static_cast<int64_t>(bufferOffsetsGm.GetValue(tableIdx));
            int64_t tableEnd = static_cast<int64_t>(bufferOffsetsGm.GetValue(tableIdx + 1));
            T embSize = embSizesGm.GetValue(tableIdx);
            int64_t embSizeI64 = static_cast<int64_t>(embSize);

            // 本核在此表中的处理范围（可能只覆盖表的一部分）
            int64_t rangeStart = (pos > tableStart) ? pos : tableStart;
            int64_t rangeEnd = (coreEnd < tableEnd) ? coreEnd : tableEnd;

            if (rangeStart >= rangeEnd) {
                tableIdx++;
                continue;
            }

            int64_t rowInTableStart = rangeStart - tableStart;
            int64_t rowInTableEnd = rangeEnd - tableStart;

            // Phase 1: 有效行 (rowInTable < embSize) → value = rowInTable
            int64_t phase1Start = rowInTableStart;
            int64_t phase1End = (embSizeI64 < rowInTableEnd) ? embSizeI64 : rowInTableEnd;
            if (phase1End < phase1Start) {
                phase1End = phase1Start;
            }

            int64_t curRow = phase1Start;
            while (curRow < phase1End) {
                int64_t batchSize = lutSize;
                if (curRow + batchSize > phase1End) {
                    batchSize = phase1End - curRow;
                }

                LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
                Adds(outLocal, lutLT, static_cast<T>(curRow), static_cast<int32_t>(batchSize));
                outQueue.EnQue(outLocal);

                LocalTensor<T> writeLocal = outQueue.DeQue<T>();
                CpLocal2Gm(addressLookupsGm[tableStart + curRow], writeLocal, batchSize);
                outQueue.FreeTensor(writeLocal);

                curRow += batchSize;
            }

            // Phase 2: 清零行 (rowInTable >= embSize) → value = 0
            int64_t phase2Start = (phase1End > rowInTableStart) ? phase1End : rowInTableStart;
            int64_t phase2End = rowInTableEnd;

            curRow = phase2Start;
            while (curRow < phase2End) {
                int64_t batchSize = lutSize;
                if (curRow + batchSize > phase2End) {
                    batchSize = phase2End - curRow;
                }

                LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
                Duplicate(outLocal, static_cast<T>(0), static_cast<int32_t>(batchSize));
                outQueue.EnQue(outLocal);

                LocalTensor<T> writeLocal = outQueue.DeQue<T>();
                CpLocal2Gm(addressLookupsGm[tableStart + curRow], writeLocal, batchSize);
                outQueue.FreeTensor(writeLocal);

                curRow += batchSize;
            }

            pos = rangeEnd;
            tableIdx++;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> lutBuf;  // 向量查找表缓冲区

    GlobalTensor<int64_t> bufferOffsetsGm;
    GlobalTensor<T> embSizesGm;
    GlobalTensor<T> addressLookupsGm;

    int64_t numTables;
    int64_t totalRows;
    int64_t lutSize;  // 向量查找表大小
    int64_t coreNum;
    bool needProcess;
};

}  // namespace InitAddressLookup

extern "C" __global__ __aicore__ void init_address_lookup(GM_ADDR bufferOffsets, GM_ADDR embSizes,
                                                          GM_ADDR addressLookups, GM_ADDR addressLookupsOut,
                                                          GM_ADDR usrWorkspace, GM_ADDR tiling)
{
    InitAddressLookup::KernelInitAddressLookup<DTYPE_EMB_SIZES> op;
    op.Init(bufferOffsets, embSizes, addressLookups, addressLookupsOut, tiling);
    op.Process();
}
