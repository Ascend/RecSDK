#include "kernel_operator.h"
using namespace AscendC;
template <typename T>
class KernelEimtable_update
{
public:
  __aicore__ inline KernelEimtable_update()
  {
  }
  __aicore__ inline void Init(GM_ADDR address, GM_ADDR embedding, GM_ADDR y)
  {
    NeedComputeAddrLen = SingleCoreAddrLen;
    if (block_idx == block_num - 1)
    {
      NeedComputeAddrLen = addrNums * sizeof(int64_t) - SingleCoreAddrLen * (block_num - 1);
    }
    round = NeedComputeAddrLen / (roundSize * sizeof(int64_t));

    pipe.InitBuffer(tbuf, roundSize * sizeof(int64_t));
    pipe.InitBuffer(inQueue, PingpongNum, Veclen);
    pipe.InitBuffer(outQueue, PingpongNum, Veclen);
    // get start index for current core, core parallel block_indx block_dim
    srcAddrGlobal.SetGlobalBuffer((__gm__ int64_t *)(address + block_idx * SingleCoreAddrLen));
    srcDataBufferGm.SetGlobalBuffer((__gm__ T *)(embedding + block_idx * SingleCoreAddrLen / sizeof(int64_t) * sizeof(T) * dim));
    outDataGm.SetGlobalBuffer((__gm__ T *)(y));
  }

  __aicore__ inline void Init_param(GM_ADDR tiling)
  {
    GET_TILING_DATA(constData, tiling);
    // 数据的维度数
    dim = constData.update_dim;
    int32_t block_total_nums = block_num;
    updateType = constData.update_type;
    addrNums = constData.addr_nums;

    // 缓冲区数量
    PingpongNum = constData.ping_pong_num;
    singleDataSize = constData.single_data_size;
    onceMoveNums = constData.once_move_nums;
    roundSize = constData.addr_max_num;

    int singlenum = (int)(addrNums / block_total_nums);
    if (singlenum % 4)
    {
      singlenum -= singlenum % 4;
    }

    Veclen = roundSize * singleDataSize * onceMoveNums;
    SingleCoreAddrLen = singlenum * sizeof(int64_t);
    cache = roundSize;
  }

  __aicore__ inline void Process()
  {

    LocalTensor<int64_t> srcAddrLocal = tbuf.Get<int64_t>(roundSize);

    int unprocess = (NeedComputeAddrLen / sizeof(int64_t)) % roundSize;

    if (round > 0)
    {
      for (int32_t i = 0; i < round; i++)
      {
        DataCopy(srcAddrLocal, srcAddrGlobal[i * roundSize], roundSize);
        MoveProcess(srcAddrLocal, i, roundSize);
      }
    }

    if (unprocess)
    {
      int unprocessOnceCopyaddr = unprocess;
      if (unprocessOnceCopyaddr % 4 != 0)
      {
          unprocessOnceCopyaddr += (4 - unprocess % 4);
      }

      DataCopy(srcAddrLocal, srcAddrGlobal[round * roundSize], unprocessOnceCopyaddr);
      MoveProcess(srcAddrLocal, round, unprocess);
    }
  }

private:
  __aicore__ inline void MoveProcess(const LocalTensor<int64_t> srcAddrLocal, const int turns, int sizes)
  {
    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0);
    LocalTensor<T> dataLocal;
    int out_index = 0;
    int offset = 0;
    int64_t address = 0;
    if (dim == onceMoveNums)
    {
      dataLocal = inQueue.AllocTensor<T>();
      DataCopy(dataLocal, srcDataBufferGm[turns * roundSize * dim], sizes * onceMoveNums);
      inQueue.EnQue(dataLocal);
      Compute(sizes);
      LocalTensor<T> dstLocal = outQueue.DeQue<T>();
      if (updateType == 0)
      {
        SetAtomicAdd<T>();
      }
      for (int i = 0; i < sizes; i++)
      {
        address = srcAddrLocal.GetValue(i);
        if (address != 0)
        {
          dstDataGm.SetGlobalBuffer((__gm__ T*)(address));
          DataCopy(dstDataGm, dstLocal[i*onceMoveNums], onceMoveNums);
        }
      }
      if (updateType == 0)
      {
        SetAtomicNone();
      }
      outQueue.FreeTensor(dstLocal);
    }
    else
    {
      for (int i = 0; i < sizes; i++)
      {
        dataLocal = inQueue.AllocTensor<T>();
        DataCopy(dataLocal, srcDataBufferGm[i * dim + turns * roundSize * dim], onceMoveNums);
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
    copyparams.blockLen = onceMoveNums * sizeof(T) * nums / 32;
    DataCopy(dstLocal, srcLocal, copyparams);
    outQueue.EnQue<T>(dstLocal);
    inQueue.FreeTensor(srcLocal);
  }

  __aicore__ inline void CopyOut(const int64_t address, const int64_t turns, const int64_t index)
  {
    LocalTensor<T> dstLocal = outQueue.DeQue<T>();

    int offset = block_idx * dim * SingleCoreAddrLen / sizeof(int64_t) + (turns * roundSize * dim) + dim * index;

    if (address != 0)
    {
      dstDataGm.SetGlobalBuffer((__gm__ T *)(address));

      if (updateType == 0)
      {
        SetAtomicAdd<T>();
      }

#if defined(__DAV_C220_VEC__)
      if (singleDataSize == 4)
      {

        copy_ubuf_to_gm_align_b32((__gm__ T *)dstDataGm.GetPhyAddr(), (__ubuf__ T *)dstLocal.GetPhyAddr(), 0,
                                  1, dim * sizeof(T), 0, 0, 0, 0);
      }
      else if (singleDataSize == 2)
      {
        copy_ubuf_to_gm_align_b16((__gm__ T *)dstDataGm.GetPhyAddr(), (__ubuf__ T *)dstLocal.GetPhyAddr(), 0,
                                  1, dim * sizeof(T), 0, 0, 0, 0);
      }
#else
      DataCopy(dstDataGm, dstLocal, onceMoveNums);
#endif
    }
    if (updateType == 0)
    {
      SetAtomicNone();
    }
    outQueue.FreeTensor(dstLocal);
  }

public:
  int32_t roundSize, round, SingleCoreAddrLen, NeedComputeAddrLen, addrNums, cache, Veclen, dim, PingpongNum;
  int32_t onceMoveNums, singleDataSize, updateType;

private:
  TPipe pipe;
  TBuf<QuePosition::LCM> tbuf;
  TQue<QuePosition::VECIN, 1> inQueue;
  TQue<QuePosition::VECOUT, 1> outQueue;
  GlobalTensor<T> srcDataBufferGm, dstDataGm, outDataGm;
  GlobalTensor<int64_t> srcAddrGlobal;
};

extern "C" __global__ __aicore__ void embedding_update_by_address(GM_ADDR address, GM_ADDR embedding, GM_ADDR y,
                                                                  GM_ADDR usrWorkspace, GM_ADDR tiling)
{
  GET_TILING_DATA(constData, tiling);

  int32_t embbedingType = constData.embbeding_type;

  switch (embbedingType)
  {
  case 0:
  {
    KernelEimtable_update<int32_t> op;
    op.Init_param(tiling);
    op.Init(address, embedding, y);
    op.Process();
  }
  break;
  case 2:
  {
    KernelEimtable_update<half> op;
    op.Init_param(tiling);
    op.Init(address, embedding, y);
    op.Process();
  }
  break;
  default:
  {
    KernelEimtable_update<float> op;
    op.Init_param(tiling);
    op.Init(address, embedding, y);
    op.Process();
  }
  break;
  }
}
