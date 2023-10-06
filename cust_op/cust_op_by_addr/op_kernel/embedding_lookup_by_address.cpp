
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
      NeedComputeAddrLen = addr_nums * sizeof(int64_t) - SingleCoreAddrLen * (block_num - 1);
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
    int32_t update_dim = constData.update_dim;
    int32_t embbeding_type = constData.embbeding_type;
    int32_t block_total_nums = block_num;
    int32_t ub_limit = constData.ub_limit;
    addr_nums = constData.addr_nums;
    if (embbeding_type == 2)
    {
      singleDataSize = 2;
    }
    else
    {
      singleDataSize = 4;
    }
    // 缓冲区数量
    PingpongNum = 1;
    int min_move_num = 32 / singleDataSize;
    // onceMoveNums表示每个数据维度需要移动的次数，(update_dim - 1 + min_move_num) / min_move_num表示除以min_move_num向下取整
    onceMoveNums = min_move_num * ((int)(update_dim - 1 + min_move_num) / min_move_num);
    int num_to_move = (int32_t)(update_dim - 1 + onceMoveNums) / onceMoveNums;
    // 每个地址需要占用sizeof(int64_t)个字节，singleDataSize表示每个数据的字节数，需要使用2倍的内存空间，因为每次移动都需要复制一份数据
    int occupyAddressBytesNum = sizeof(int64_t) + singleDataSize * onceMoveNums * num_to_move * PingpongNum * 2;
    // 计算一轮计算中最多计算多少个addr，最后的 /4 再*4 是为了与32对齐，因为sizeof(int64_t) = 8
    int addrMaxNum = ((int)((int)(ub_limit / occupyAddressBytesNum) / 4)) * 4;
    int singlenum = (int)(addr_nums / block_total_nums);
    if (singlenum % 4)
    {
      singlenum -= singlenum % 4;
    }
    roundSize = addrMaxNum;
    Veclen = roundSize * singleDataSize * onceMoveNums;
    SingleCoreAddrLen = singlenum * sizeof(int64_t);
    cache = roundSize;
    dim = update_dim;
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

    int unprocess = (NeedComputeAddrLen / sizeof(int64_t)) % roundSize;
    if (unprocess)
    {
      // 处理 addresslist 不对齐32b
      int unprocess_once_copyaddr = unprocess;
      if (unprocess_once_copyaddr % 4 != 0)
      {
        unprocess_once_copyaddr += (4 - unprocess % 4);
      }

      DataCopy(srcAddrLocal, srcAddrGlobal[round * roundSize], unprocess_once_copyaddr);
      MoveProcess(srcAddrLocal, round, unprocess);
    }
  }

private:
  __aicore__ inline void MoveProcess(const LocalTensor<int64_t> srcAddrLocal, const int turns, int sizes)
  {
    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0);
    LocalTensor<T> dataLocal;
    bool isFull = true;
    int nums = 0;
    int out_index = 0;
    int times = onceMoveNums / 8;
    int tmp_cache = cache - 1;

    for (int i = 0; i < sizes; i++)
    {

      dataLocal = inQueue.AllocTensor<T>();
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
        inQueue.EnQue(dataLocal);
        Compute(1);
        CopyOut(i, turns, 1);
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
  int32_t addr_nums;
  int32_t onceMoveNums, singleDataSize, update_type;

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

  int32_t embbeding_type = constData.embbeding_type;

  switch (embbeding_type)
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
