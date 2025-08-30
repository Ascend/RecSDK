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

#include "kernel_operator.h"
using namespace AscendC;

namespace AscendC {

constexpr int32_t SIZE_OF_HALF = 2;
constexpr int32_t SIZE_OF_FLOAT_OR_INT = 4;
constexpr int32_t PADDING_ZERO_NUM_PER_TIME = 8;

template <typename T>
class KernelEimtable
{
public:
    __aicore__ inline KernelEimtable()
    {
    }
    __aicore__ inline void Init(GM_ADDR address, GM_ADDR y)
    {
        needComputeAddrLen = singleCoreAddrLen;
        if (block_idx == block_num - 1) // 最后一个core,需要多计算的addr长度
        {
            needComputeAddrLen = addrNums * sizeof(int64_t) - singleCoreAddrLen * (block_num - 1);
        }
        loopCount = needComputeAddrLen / (addrNumPerLoop * sizeof(int64_t)); // 可能为0

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(tbuf, addrNumPerLoop * sizeof(int64_t));

        pipe.InitBuffer(inQueue, pingpongNum, veclen);
        pipe.InitBuffer(outQueue, pingpongNum, veclen);

#ifdef L2_CACHE_HINT
        // set `GlobalTensor` cache mode explicitly
    srcAddrGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    dstDataGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
#endif

        // get start index for current core, core parallel block_indx block_dim，即使是最后一个核也应该多初始化一些，并对齐4的倍数
        srcAddrGlobal.SetGlobalBuffer((__gm__ int64_t *)(address + block_idx * singleCoreAddrLen), needComputeAddrLen);
        dstDataGm.SetGlobalBuffer((__gm__ T *)(y));
    }

    __aicore__ inline void Init_param(GM_ADDR tiling)
    {
        GET_TILING_DATA(constData, tiling);

        pingpongNum = constData.ping_pong_num;
        addrNums = constData.addr_nums;
        dim = constData.embedding_dim;
        addrNumPerLoop = constData.addr_per_loop;
        typeSize = constData.type_size;
        embDimAligned = constData.emb_dim_aligned;

        int singleCoreAddrNum = (int)(addrNums / block_num); // 有可能没有整除，最后的核会处理更多的数据
        singleCoreAddrNum = singleCoreAddrNum & (~3); // & (~3) 代表取4的倍数向下取整，处理的地址占8字节，对齐32B的话，数量需要是4倍数

        singleCoreAddrLen = singleCoreAddrNum * sizeof(int64_t);
        veclen = addrNumPerLoop * typeSize * embDimAligned;  // 向上对齐32B
        cache = constData.addr_per_loop;
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int64_t> srcAddrLocal = tbuf.Get<int64_t>(addrNumPerLoop);

        if (loopCount > 0)
        {
            for (int32_t i = 0; i < loopCount; i++) {
                DataCopy(srcAddrLocal, srcAddrGlobal[i * addrNumPerLoop], addrNumPerLoop);
                MoveProcess(srcAddrLocal, i, addrNumPerLoop);
            }
        }
        // 处理最后一张卡剩下的addr
        int unProcess = (needComputeAddrLen / sizeof(int64_t)) % addrNumPerLoop;
        if (unProcess)
        {
            int unProcessAligned = static_cast<int>
            ((static_cast<unsigned int>(unProcess) + 3) & (~3U)); // 处理 addressList 不对齐32b的情况
            // 地址列表访问越界，对齐考虑无问题，会自动多申请一部分，兼容
            DataCopy(srcAddrLocal, srcAddrGlobal[loopCount * addrNumPerLoop], unProcessAligned);
            MoveProcess(srcAddrLocal, loopCount, unProcess);
        }
    }

public:
    int32_t addrNumPerLoop, loopCount, singleCoreAddrLen, needComputeAddrLen, veclen, dim, pingpongNum, cache;
    int32_t addrNums;
    int32_t embDimAligned, typeSize, updateType;

private:
    __aicore__ inline void MoveProcess(const LocalTensor<int64_t> srcAddrLocal, const int turns, int addrNum)
    {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        LocalTensor<T> dataLocal = inQueue.AllocTensor<T>(); // Queue的大小可以容下一个循环的所有emb
        bool isFull = false;
        int nums = 0;
        int outIndex = 0;
        int times = embDimAligned >> 3; // >>3位运算：除以8。 embDimAligned一定是8的倍数，若地址无效时，每次填充8个0
        int tmpCache = cache - 1; // 设计初是一次cache执行多次copyin、一次compute和一次copyout，现状是一次loop就只对应一次cache

        for (int i = 0; i < addrNum; i++) {
            // 多次copyIn， 对应一次compute和copyOut，由cache决定
            dataLocal = isFull ? inQueue.AllocTensor<T>() : dataLocal;
            int64_t address = srcAddrLocal.GetValue(i);
            if (address != 0) {
#ifdef L2_CACHE_HINT
                srcDataBufferGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
#endif
                srcDataBufferGm.SetGlobalBuffer((__gm__ T *)(address), embDimAligned);
                DataCopy(dataLocal[embDimAligned * nums], srcDataBufferGm, embDimAligned);
            } else {
                for (int j = 0; j < times; j++) {
                    Duplicate(dataLocal[embDimAligned * nums + j * PADDING_ZERO_NUM_PER_TIME],
                              (T)0, PADDING_ZERO_NUM_PER_TIME);
                }
            }

            nums++;
            isFull = (i == tmpCache || i == addrNum - 1); // cache满了，或者最后一个地址
            if (isFull) {
                inQueue.EnQue(dataLocal);
                Compute(nums);
                CopyOut(outIndex, turns, nums);
                nums = 0;
                outIndex = i + 1;
                tmpCache += cache;
            }
        }
    }

    __aicore__ inline void Compute(const int nums)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (embDimAligned * sizeof(T) * nums) >> 5; // >> 5， 除以32，ub空间对齐
        DataCopy(dstLocal, srcLocal, copyParams);

        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(const int index, const int turns, const int nums)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();

        int offset = block_idx * dim * singleCoreAddrLen /
                sizeof(int64_t) + (turns * addrNumPerLoop * dim) + dim * index;
#if defined(__DAV_C220_VEC__)
        if (typeSize == SIZE_OF_FLOAT_OR_INT) {
            copy_ubuf_to_gm_align_b32((__gm__ T *)dstDataGm[offset].GetPhyAddr(),
                                      (__ubuf__ T *)dstLocal.GetPhyAddr(), 0, nums, dim * sizeof(T), 0, 0, 0, 0);
        } else if (typeSize == SIZE_OF_HALF) {
            copy_ubuf_to_gm_align_b16((__gm__ T *)dstDataGm[offset].GetPhyAddr(),
                                      (__ubuf__ T *)dstLocal.GetPhyAddr(), 0, nums, dim * sizeof(T), 0, 0, 0, 0);
        }
#else

        DataCopy(dstDataGm[offset], dstLocal, embDimAligned * nums);
#endif
        outQueue.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::LCM> tbuf;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    GlobalTensor<T> srcDataBufferGm, dstDataGm;
    GlobalTensor<int64_t> srcAddrGlobal;
};
}

extern "C" __global__ __aicore__ void embedding_lookup_by_address(GM_ADDR address, GM_ADDR y, GM_ADDR usrWorkspace,
                                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(constData, tiling);

    int32_t embeddingType = constData.embedding_type;

    switch (embeddingType) {
        case 0: {
                AscendC::KernelEimtable<int32_t> op;
                op.Init_param(tiling);
                op.Init(address, y);
                op.Process();
            }
            break;
        case 2: {
                AscendC::KernelEimtable<half> op;
                op.Init_param(tiling);
                op.Init(address, y);
                op.Process();
            }
            break;
        default: {
                AscendC::KernelEimtable<float> op;
                op.Init_param(tiling);
                op.Init(address, y);
                op.Process();
            }
            break;
    }
}
