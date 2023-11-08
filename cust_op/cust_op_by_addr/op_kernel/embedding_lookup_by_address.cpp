#include "kernel_operator.h"
using namespace AscendC;
template <typename T>
class KernelEimtable
{
public:
  __aicore__ inline KernelEimtable()
  {
  }
  __aicore__ inline void Init(GM_ADDR address, GM_ADDR y)
  {

    NeedComputeAddrLen = SingleCoreAddrLen;
    if (block_idx == block_num - 1)
    {
      NeedComputeAddrLen = addrNums * sizeof(int64_t) - SingleCoreAddrLen * (block_num - 1);
    }
    round = NeedComputeAddrLen / (roundSize * sizeof(int64_t));
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(tbuf, roundSize * sizeof(int64_t));

    pipe.InitBuffer(inQueue, PingpongNum, Veclen);
    pipe.InitBuffer(outQueue, PingpongNum, Veclen); //

    // get start index for current core, core parallel block_indx block_dim
    srcAddrGlobal.SetGlobalBuffer((__gm__ int64_t *)(address + block_idx * SingleCoreAddrLen));
    dstDataGm.SetGlobalBuffer((__gm__ T *)(y));
  }

  __aicore__ inline void Init_param(GM_ADDR tiling)
  {
    GET_TILING_DATA(constData, tiling);
    // 数据的维度数
    dim = constData.update_dim;
    int32_t blockTotalNums = block_num;
    addrNums = constData.addr_nums;
    // 缓冲区数量
    PingpongNum = constData.ping_pong_num;
    singleDataSize = constData.single_data_size;
    onceMoveNums = constData.once_move_nums;
    roundSize = constData.addr_max_num;

    int singleNum = (int)(addrNums / blockTotalNums);
    if (singleNum % 4)
    {
      singleNum -= singleNum % 4;
    }

    Veclen = roundSize * singleDataSize * onceMoveNums;
    SingleCoreAddrLen = singleNum * sizeof(int64_t);
    cache = roundSize;
  }

  __aicore__ inline void Process()
  {

    LocalTensor<int64_t> srcAddrLocal = tbuf.Get<int64_t>(roundSize);

    if (round > 0)
    {
      for (int32_t i = 0; i < round; i++)
      {
        DataCopy(srcAddrLocal, srcAddrGlobal[i * roundSize], roundSize);
        MoveProcess(srcAddrLocal, i, roundSize);
      }
    }

    int unProcess = (NeedComputeAddrLen / sizeof(int64_t)) % roundSize;
    if (unProcess)
    {
      // 处理 addresslist 不对齐32b
      int unProcessOnceCopyAddr = unProcess;
      if (unProcessOnceCopyAddr % 4 != 0)
      {
          unProcessOnceCopyAddr += (4 - unProcess % 4);
      }

      DataCopy(srcAddrLocal, srcAddrGlobal[round * roundSize], unProcessOnceCopyAddr);
      MoveProcess(srcAddrLocal, round, unProcess);
    }
  }

private:
  __aicore__ inline void MoveProcess(const LocalTensor<int64_t> srcAddrLocal, const int turns, int sizes)
  {
    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0);
    LocalTensor<T> dataLocal = inQueue.AllocTensor<T>();
    bool isFull = false;
    int nums = 0;
    int outIndex = 0;
    int times = onceMoveNums / 8;
    int tmpCache = cache - 1;

    for (int i = 0; i < sizes; i++)
    {

      dataLocal = isFull ? inQueue.AllocTensor<T>() : dataLocal;
      int64_t address = srcAddrLocal.GetValue(i);

      if (address != 0)
      {
        srcDataBufferGm.SetGlobalBuffer((__gm__ T *)(address));
        DataCopy(dataLocal[onceMoveNums * nums], srcDataBufferGm, onceMoveNums);
      }
      else
      {

        for (int j = 0; j < times; j++)
        {
          Duplicate(dataLocal[onceMoveNums * nums + j * 8], (T)0, 8);
        }

      }

      nums++;
      isFull = (i == tmpCache || i == sizes - 1);
      if (isFull)
      {
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
    copyParams.blockLen = onceMoveNums * sizeof(T) * nums / 32;
    DataCopy(dstLocal, srcLocal, copyParams);

    outQueue.EnQue<T>(dstLocal);
    inQueue.FreeTensor(srcLocal);
  }

  __aicore__ inline void CopyOut(const int index, const int turns, const int nums)
  {
    LocalTensor<T> dstLocal = outQueue.DeQue<T>();

    int offset = block_idx * dim * SingleCoreAddrLen / sizeof(int64_t) + (turns * roundSize * dim) + dim * index;
#if defined(__DAV_C220_VEC__)
    if (singleDataSize == 4)
    {
      copy_ubuf_to_gm_align_b32((__gm__ T *)dstDataGm[offset].GetPhyAddr(), (__ubuf__ T *)dstLocal.GetPhyAddr(), 0,
                                nums, dim * sizeof(T), 0, 0, 0, 0);
    }
    else if (singleDataSize == 2)
    {
      copy_ubuf_to_gm_align_b16((__gm__ T *)dstDataGm[offset].GetPhyAddr(), (__ubuf__ T *)dstLocal.GetPhyAddr(), 0,
                                nums, dim * sizeof(T), 0, 0, 0, 0);
    }
#else

    DataCopy(dstDataGm[offset], dstLocal, onceMoveNums * nums);
#endif
    outQueue.FreeTensor(dstLocal);
  }

public:
  int32_t roundSize, round, SingleCoreAddrLen, NeedComputeAddrLen, cache, Veclen, dim, PingpongNum;
  int32_t addrNums;
  int32_t onceMoveNums, singleDataSize, updateType;

private:
  TPipe pipe;
  TBuf<QuePosition::LCM> tbuf;
  TQue<QuePosition::VECIN, 1> inQueue;
  TQue<QuePosition::VECOUT, 1> outQueue;
  GlobalTensor<T> srcDataBufferGm, dstDataGm, outDataGm;
  GlobalTensor<int64_t> srcAddrGlobal;
};

extern "C" __global__ __aicore__ void embedding_lookup_by_address(GM_ADDR address, GM_ADDR y, GM_ADDR usrWorkspace,
                                                                  GM_ADDR tiling)
{
  GET_TILING_DATA(constData, tiling);

  int32_t embeddingType = constData.embbeding_type;

  switch (embeddingType)
  {
  case 0:
  {
    KernelEimtable<int32_t> op;
    op.Init_param(tiling);
    op.Init(address, y);
    op.Process();
  }
  break;
  case 2:
  {
    KernelEimtable<half> op;
    op.Init_param(tiling);
    op.Init(address, y);
    op.Process();
  }
  break;
  default:
  {
    KernelEimtable<float> op;
    op.Init_param(tiling);
    op.Init(address, y);
    op.Process();
  }
  break;
  }
}
