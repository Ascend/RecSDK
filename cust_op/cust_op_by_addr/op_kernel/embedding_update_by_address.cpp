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

namespace KernelOps {
constexpr int32_t SIZE_OF_HALF = 2;
constexpr int32_t SIZE_OF_FLOAT_OR_INT = 4;

template <typename T>
class KernelEimtable_update
{
public:
  __aicore__ inline KernelEimtable_update()
  {
  }
  __aicore__ inline void Init(GM_ADDR address, GM_ADDR embedding, GM_ADDR y)
  {
    needComputeAddrLen = singleCoreAddrLen;
    if (block_idx == block_num - 1)
    {
        needComputeAddrLen = addrNums * sizeof(int64_t) - singleCoreAddrLen * (block_num - 1);
    }
    loopCount = needComputeAddrLen / (addrNumPerLoop * sizeof(int64_t));

    pipe.InitBuffer(tbuf, addrNumPerLoop * sizeof(int64_t));
    pipe.InitBuffer(inQueue, pingpongNum, veclen);
    pipe.InitBuffer(outQueue, pingpongNum, veclen);

    // set `GlobalTensor` cache mode explicitly
    srcAddrGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    srcDataBufferGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL); 
    outDataGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);

    // get start index for current core, core parallel block_indx block_dim
    srcAddrGlobal.SetGlobalBuffer((__gm__ int64_t *)(address + block_idx * singleCoreAddrLen));
    srcDataBufferGm.SetGlobalBuffer((__gm__ T *)(embedding + block_idx * singleCoreAddrLen
    / sizeof(int64_t) * sizeof(T) * dim));
    outDataGm.SetGlobalBuffer((__gm__ T *)(y));
  }

  __aicore__ inline void Init_param(GM_ADDR tiling)
  {
    GET_TILING_DATA(constData, tiling);

    pingpongNum = constData.ping_pong_num;
    dim = constData.update_dim;
    updateType = constData.update_type;
    addrNums = constData.addr_nums;
    typeSize = constData.type_size;
    inputDimAligned = constData.input_dim_aligned;
    addrNumPerLoop = constData.addr_per_loop;

    int singleCoreAddrNum = (int)(addrNums / block_num);
    singleCoreAddrNum = singleCoreAddrNum & (~3); // & (~3) 代表取4的倍数向下取整，处理的地址占8字节，对齐32B的话，数量需要是4倍数

    veclen = addrNumPerLoop * typeSize * inputDimAligned;
    singleCoreAddrLen = singleCoreAddrNum * sizeof(int64_t);
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

    int unProcess = (needComputeAddrLen / sizeof(int64_t)) % addrNumPerLoop;
    if (unProcess)
    {
        int unProcessAligned = (static_cast<unsigned int>(unProcess) + 3) & (~3U); // 处理 addressList 不对齐32b的情况
        DataCopy(srcAddrLocal, srcAddrGlobal[loopCount * addrNumPerLoop], unProcessAligned);
        MoveProcess(srcAddrLocal, loopCount, unProcess);
    }
  }

private:
    __aicore__ inline void MoveProcess(const LocalTensor<int64_t> srcAddrLocal, const int turns, int addrNum)
    {
        set_flag(PIPE_MTE2, PIPE_S, 0);
        wait_flag(PIPE_MTE2, PIPE_S, 0);
        LocalTensor<T> dataLocal;

        int64_t address = 0;
        if (dim == inputDimAligned) // copyIn 和 compute一次，copyOut多次
        {
            dataLocal = inQueue.AllocTensor<T>();
            DataCopy(dataLocal, srcDataBufferGm[turns * addrNumPerLoop * dim], addrNum * inputDimAligned);
            inQueue.EnQue(dataLocal);

            Compute(addrNum); // 只有copyOut的管道支持拷贝到gm上

            LocalTensor<T> dstLocal = outQueue.DeQue<T>();
            if (updateType == 0) {
                SetAtomicAdd<T>();
            }
            for (int i = 0; i < addrNum; i++) {
                address = srcAddrLocal.GetValue(i);
                if (address != 0) {
                    dstDataGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
                    dstDataGm.SetGlobalBuffer((__gm__ T*)(address));
                    DataCopy(dstDataGm, dstLocal[i * inputDimAligned], inputDimAligned);
                }
            }
            if (updateType == 0) {
                SetAtomicNone();
            }
            outQueue.FreeTensor(dstLocal);
        } else {
            for (int i = 0; i < addrNum; i++) {
                dataLocal = inQueue.AllocTensor<T>();
                DataCopy(dataLocal, srcDataBufferGm[i * dim + turns * addrNumPerLoop * dim], inputDimAligned);
                inQueue.EnQue<T>(dataLocal);
                Compute(1);
                address = srcAddrLocal.GetValue(i);
                CopyOut(address, turns, i);
            }
        }
    }

    __aicore__ inline void Compute(const int nums)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        DataCopyParams copyparams;
        copyparams.blockCount = 1;
        copyparams.blockLen = (inputDimAligned * sizeof(T) * nums) >> 5; // >> 5， 除以32，ub空间对齐
        DataCopy(dstLocal, srcLocal, copyparams);
        outQueue.EnQue<T>(dstLocal);
        inQueue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(const int64_t address, const int64_t turns, const int64_t index)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();

        if (address != 0) {
            dstDataGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
            dstDataGm.SetGlobalBuffer((__gm__ T *)(address));

            if (updateType == 0) {
                SetAtomicAdd<T>();
            }

#if defined(__DAV_C220_VEC__)
            if (typeSize == SIZE_OF_FLOAT_OR_INT) {
                copy_ubuf_to_gm_align_b32((__gm__ T *)dstDataGm.GetPhyAddr(), (__ubuf__ T *)dstLocal.GetPhyAddr(), 0,
                                          1, dim * sizeof(T), 0, 0, 0, 0);
            } else if (typeSize == SIZE_OF_HALF) {
                copy_ubuf_to_gm_align_b16((__gm__ T *)dstDataGm.GetPhyAddr(), (__ubuf__ T *)dstLocal.GetPhyAddr(), 0,
                                          1, dim * sizeof(T), 0, 0, 0, 0);
            }
#else
            DataCopy(dstDataGm, dstLocal, inputDimAligned);
#endif
        }
        if (updateType == 0) {
            SetAtomicNone();
        }
        outQueue.FreeTensor(dstLocal);
    }

public:
  int32_t addrNumPerLoop, loopCount, singleCoreAddrLen, needComputeAddrLen, addrNums, cache, veclen, dim, pingpongNum;
  int32_t inputDimAligned, typeSize, updateType;

private:
  TPipe pipe;
  TBuf<QuePosition::LCM> tbuf;
  TQue<QuePosition::VECIN, 1> inQueue;
  TQue<QuePosition::VECOUT, 1> outQueue;
  GlobalTensor<T> srcDataBufferGm, dstDataGm, outDataGm;
  GlobalTensor<int64_t> srcAddrGlobal;
};
}

extern "C" __global__ __aicore__ void embedding_update_by_address(GM_ADDR address, GM_ADDR embedding, GM_ADDR y,
                                                                  GM_ADDR usrWorkspace, GM_ADDR tiling)
{
  GET_TILING_DATA(constData, tiling);

  int32_t embeddingType = constData.embedding_type;

  switch (embeddingType)
  {
  case 0:
  {
    KernelOps::KernelEimtable_update<int32_t> op;
    op.Init_param(tiling);
    op.Init(address, embedding, y);
    op.Process();
  }
  break;
  case 2:
  {
    KernelOps::KernelEimtable_update<half> op;
    op.Init_param(tiling);
    op.Init(address, embedding, y);
    op.Process();
  }
  break;
  default:
  {
    KernelOps::KernelEimtable_update<float> op;
    op.Init_param(tiling);
    op.Init(address, embedding, y);
    op.Process();
  }
  break;
  }
}
