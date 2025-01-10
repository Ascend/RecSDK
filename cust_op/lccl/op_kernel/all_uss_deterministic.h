/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LCCL_ALL_USS_DETERMINISTIC_H
#define LCCL_ALL_USS_DETERMINISTIC_H

#include "collectives.h"
#include "ipc_queue.h"

using namespace AscendC;

template<typename T>
class AllUssDeterministic : public Collectives {
    constexpr static int INVALID_RANK_NUM = 0xFFFFFFFF;  // 非法rank
    constexpr static int64_t SHARE_QUE_DEPTH = 16;  // 单个共享队列深度
    constexpr static int64_t MULTI_RANK_SIZE = 32;
    constexpr static int64_t MAX_FLAG_OFFSET = 128;

    constexpr static int64_t IDLER_CORE = 0;  // 闲置的核
    constexpr static int64_t PRODUCER_CORE = 1;  // 生产组，负责向共享内存写入数据，input->share，或者share->share
    constexpr static int64_t CONSUMER_CORE = 2;  // 消费组，负责从共享内存读出数据，share->output

    constexpr static int64_t SYNC_FLAG_START = 256; // 用于确定性计算的同步符号位置

public:
    __aicore__ inline AllUssDeterministic(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag)
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR send_count_matrix, GM_ADDR shape_vec, GM_ADDR peer_mem,
                                GM_ADDR restore, GM_ADDR output, int64_t rank, int64_t rankSize,
                                int64_t magic, int64_t dim, int64_t outShape)
    {
        this->root = 0;
        this->len = 0;
        this->magic = magic;
        this->rank = rank;
        this->rankSize = rankSize;
        this->dim = dim;
        this->restorePtr = restore;
        this->outputPtr = output;
        // max 16 core for each process stage due to hardware only has max 32 core
        this->coreNumsPerStage = rankSize < 16 ? rankSize : 16;
        
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
        
        GlobalTensor<int64_t> peerMemsAddrGm;
        int64_t peer_mem_addr = reinterpret_cast<int64_t>(peer_mem);
        peerMemsAddrGm.SetGlobalBuffer((__gm__ int64_t*)peer_mem_addr, rankSize * sizeof (int64_t));
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = (GM_ADDR)(peerMemsAddrGm.GetValue(i))+
                            (this->magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }

        sync.Init(rank, rankSize, shareAddrs, blockIdx, blockNum);

        sendCountMatrixGm.SetGlobalBuffer((__gm__ int64_t*)send_count_matrix, rankSize * rankSize * sizeof (int64_t));
        // 初始化共享内存信息
        InitShare();
        // 初始化核分组
        InitCoreGroup();
        // 初始化数据切片
        InitDataSlice();

        // 初始化输入输出
        for (int j = 0; j < rankSize; j++) {
            sendLen += sendCountMatrixGm.GetValue(rank * rankSize + j);
        }
        inputGt.SetGlobalBuffer((__gm__ T*)input, sendLen * sizeof(T));
        for (int j = 0; j < rankSize; j++) {
            revLen += sendCountMatrixGm.GetValue(j * rankSize + rank);
        }
        pipe.InitBuffer(tempBuffer, PING_PONG_SIZE, UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE);
        outputGt.SetGlobalBuffer((__gm__ T*)output, outShape * dim * sizeof(T));

        int initSize = outShape * dim / coreNumsPerStage / PING_PONG_SIZE;
        outputGtInit.SetGlobalBuffer((__gm__ T*)output + initSize * blockIdx);
        sync.SetInnerFlag(magic, 0, rank, blockIdx + MAX_FLAG_OFFSET);
        pipe_barrier(PIPE_ALL);
        if (blockIdx < (coreNumsPerStage * PING_PONG_SIZE)) {
            AscendC::InitGlobalMemory(outputGtInit, initSize * sizeof(T), (T)(0));
            sync.SetInnerFlag(magic, 1, rank, blockIdx + MAX_FLAG_OFFSET);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreGroup == PRODUCER_CORE) {
            if (blockIdx != 0) {
                sync.WaitInnerFlag(1, 0, rank, SYNC_FLAG_START + blockIdx);
                sync.SetInnerFlag(0, 0, rank, SYNC_FLAG_START + blockIdx);
            } else {
                sync.SetInnerFlag(1, 0, rank, SYNC_FLAG_START + blockIdx);
            }
            ProducerStage();
        }
        if (coreGroup == CONSUMER_CORE) {
            for (int i = 0; i < coreNumsPerStage; i++) {
                sync.WaitInnerFlag(magic, 1, rank, i + MAX_FLAG_OFFSET);
            }
            sync.WaitInnerFlag(1, 0, rank, SYNC_FLAG_START + blockIdx - coreNumsPerStage);
            ConsumerStage();
            sync.SetInnerFlag(1, 0, rank, SYNC_FLAG_START + blockIdx - coreNumsPerStage + 1);
        }
    }

private:
    // 计算rank数量较大时的queNum 以及  每个队列里单块可放入的元素数量queElemLen
    __aicore__ inline void InitShare()
    {
        int64_t queNum = coreNumsPerStage;  // 共享内存最小切分数量为单阶段核数
        if (rankSize > coreNumsPerStage) {
            queNum = rankSize;
        }
        queElemLen = IPC_BUFF_MAX_SIZE / sizeof(T) / queNum / SHARE_QUE_DEPTH;  // 计算共享队列元素大小
    }

    __aicore__ inline void InitCoreGroup()
    {
        // 每个rank在每个stage分到的core数量， 多卡下为1
        coreNumPerRank = coreNumsPerStage / rankSize > 1 ?
                         coreNumsPerStage / rankSize : 1;
        // 小数据量下，单rank数据仅需一个核
        if (len < queElemLen) {
            coreNumPerRank = 1;
        }
        // 多卡下为coreNumsPerStage
        coreNumPerStage = coreNumPerRank * rankSize < coreNumsPerStage ?
                          coreNumPerRank * rankSize : coreNumsPerStage;
        // 一个core处理多少rank
        rankNumPerCore = CeilDiv(rankSize, coreNumPerStage);

        // 单核负责多rank时，计算虚拟核索引并储存（索引为多个，由一个物理核执行其它虚拟核索引的操作）
        // 多卡场景下flagNumPerStage 为 ranksize
        flagNumPerStage = coreNumPerStage * rankNumPerCore;
        // 将core 分类到不同的stage， 并且找到本core对应要处理的rank
        if (blockIdx < coreNumPerStage) {
            coreGroup = PRODUCER_CORE;
            for (auto i = 0; i < rankNumPerCore; ++i) {
                groupCoreIdx[i] = blockIdx * rankNumPerCore + i;
            }
        } else if (blockIdx < coreNumPerStage + coreNumPerStage) {
            coreGroup = CONSUMER_CORE;
            for (auto i = 0; i < rankNumPerCore; ++i) {
                groupCoreIdx[i] = blockIdx * rankNumPerCore + i - flagNumPerStage;
            }
        } else {
            coreGroup = IDLER_CORE;
        }
    }

    __aicore__ inline void InitDataSlice()
    {
        queLen = queElemLen * SHARE_QUE_DEPTH;  // 一个que的可放入的元素数量
        queSize = queLen * sizeof(T);

        // 生产者负责搬运本rank的输入数据至共享内存，input-->share
        if (coreGroup == PRODUCER_CORE) {
            ProducerDataSlice();
        } else if (coreGroup == CONSUMER_CORE) {
            ConsumerDataSlice();
        }
    }

    __aicore__ inline void ProducerDataSlice()
    {
        maxSliceNum = 0;
        for (auto i = 0; i < rankNumPerCore; ++i) {
            // 当前核负责的rank， 因为是基于trackRankSize计算的groupCoreIdx，所以要乘RANK_SIZE_TWO
            targetRank[i] = groupCoreIdx[i] / coreNumPerRank;
            if (targetRank[i] >= rankSize) {
                targetRank[i] = INVALID_RANK_NUM;
                continue;
            }
            // 当前核负责的ipcQue
            writeQue[i].Init(&sync, magic, shareAddrs[targetRank[i]] + IPC_DATA_OFFSET +
                             (rank * coreNumPerRank + blockIdx%coreNumPerRank) * queSize, queLen, queElemLen);

            // 当前核负责的数据长度和偏移
            sendOffset[i] = 0;
            for (int j = 0; j < targetRank[i]; j++) {
                sendOffset[i] += sendCountMatrixGm.GetValue(rank * rankSize + j);
            }
            inputDataLen[i] = sendCountMatrixGm.GetValue(rank * rankSize + targetRank[i]);
            SplitData(inputDataLen[i], coreNumPerRank, groupCoreIdx[i] % coreNumPerRank, inputOffset[i],
                      inputLen[i], sendOffset[i]);
            // 当前核负责的数据切片数，能分成一个que中的多少小块
            sliceNum[i] = CeilDiv(inputLen[i], queElemLen);
            if (sliceNum[i] > maxSliceNum) {
                maxSliceNum = sliceNum[i];
            }
        }
    }

    __aicore__ inline void ConsumerDataSlice()
    {
        maxSliceNum = 0;
        for (auto i = 0; i < rankNumPerCore; ++i) {
            // 当前核负责的rank
            targetRank[i] = groupCoreIdx[i] / coreNumPerRank;
            if (targetRank[i] >= rankSize) {
                targetRank[i] = INVALID_RANK_NUM;
                continue;
            }
            // 当前核负责的ipcQue
            readQue[i].Init(
                &sync, magic,
                shareAddrs[rank] + IPC_DATA_OFFSET +
                    (targetRank[i] * coreNumPerRank + blockIdx % coreNumPerRank) * queSize,
                queLen, queElemLen
            );
            // 当前核负责的数据长度和偏移
            revOffset[i] = 0;
            for (int j = 0; j < targetRank[i]; j++) {
                revOffset[i] += sendCountMatrixGm.GetValue(j * rankSize + rank);
            }
            outputDataLen[i] = sendCountMatrixGm.GetValue(targetRank[i] * rankSize + rank);

            SplitData(outputDataLen[i], coreNumPerRank, groupCoreIdx[i] % coreNumPerRank, outputOffset[i],
                      outputLen[i], revOffset[i]);
            // 当前核负责的数据切片数
            sliceNum[i] = CeilDiv(outputLen[i], queElemLen);
            if (sliceNum[i] > maxSliceNum) {
                maxSliceNum = sliceNum[i];
            }
        }
    }

    __aicore__ inline void SplitData(const int64_t totalLen, const int64_t useCoreNum, const int64_t useCoreIdx,
                                     int64_t& dataOffset, int64_t& dataLen, int startOffset)
    {
        // 向上整除获取每个core切分的数据个数
        dataLen = CeilDiv(totalLen, useCoreNum);
        // 数据量极小或略微超过核数的情况，后面若干个core数据量为0
        dataOffset = useCoreIdx * dataLen + startOffset;  // 使用当前block在useBlock里的相对索引来计算偏移
        if (useCoreIdx * dataLen >= totalLen) {
            dataOffset = totalLen + startOffset;
            dataLen = 0;
            return;
        }
        // 非整除情况，最后一个core数据量为剩余数据量
        if (dataOffset + dataLen - startOffset > totalLen) {
            dataLen = totalLen - useCoreIdx * dataLen;
        }
    }

    __aicore__ inline void ProducerStage()
    {
        for (auto i = 0; i < rankNumPerCore; ++i) {
            if (targetRank[i] == INVALID_RANK_NUM) {
                continue;
            }

            // 写共享内存队列时，需要等待当前rank
            waitRankListForWrite[i][0] = targetRank[i];
            waitNumForWrite[i] = 1;
            waitBlockForWrite[i] = rank * coreNumPerRank + groupCoreIdx[i] % coreNumPerRank + flagNumPerStage;
        }
        InputToSharePipeline();
    }

    __aicore__ inline void InputToSharePipeline()
    {
        int64_t flagValue[MULTI_RANK_SIZE];  // 要等待标志位的储存值
        for (auto i = 0; i < rankNumPerCore; ++i) {
            flagValue[i] = -1;  // 统一赋值为-1，便于后续小于判断
        }
        // 以最多切片maxSliceNum为切片数进行循环，切片数不足的不拷贝
        for (auto sliceIdx = 0; sliceIdx < maxSliceNum; ++sliceIdx) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                if (targetRank[i] == INVALID_RANK_NUM) {
                    continue;
                }
                InputToShareSlice(i, sliceIdx, flagValue[i]);
            }
        }
    }

    __aicore__ inline void InputToShareSlice(int64_t idx, int64_t sliceIdx, int64_t& flagValue)
    {
        readGt = inputGt[sliceIdx * queElemLen + inputOffset[idx]];
        // 计算当前切片拷贝数据量，数据量为0时不拷贝
        copyLen = inputLen[idx] - queElemLen * sliceIdx;
        if (copyLen > queElemLen) {
            copyLen = queElemLen;
        } else if (copyLen < 0) {
            copyLen = 0;
        }
        writeQue[idx].DeQue(waitRankListForWrite[idx], waitNumForWrite[idx], waitBlockForWrite[idx], sliceIdx);
        writeGt = writeQue[idx].EnQue();
        if (copyLen > 0) {
            CpGM2GMPingPong<T>(copyLen * sizeof(T), readGt, writeGt, COPYONLY);
        }
        sync.SetInnerFlag(magic, sliceIdx, rank, groupCoreIdx[idx]);
    }

    __aicore__ inline void ConsumerStage()
    {
        int64_t flagValue[MULTI_RANK_SIZE];  // 要等待标志位的储存值
        for (auto i = 0; i < rankNumPerCore; ++i) {
            flagValue[i] = -1;
        }
        // 以最多切片maxSliceNum为切片数进行循环，切片数不足的不拷贝
        for (auto sliceIdx = 0; sliceIdx < maxSliceNum; ++sliceIdx) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                if (targetRank[i] == INVALID_RANK_NUM) {
                    continue;
                }
                ShareToOutputSlice(i, sliceIdx, flagValue[i]);
            }
        }
    }

    __aicore__ inline void ShareToOutputSlice(int64_t idx, int64_t sliceIdx, int64_t& flagValue)
    {
        // 计算当前切片拷贝数据量，数据量为0时不拷贝, copyLen
        copyLen = outputLen[idx] - queElemLen * sliceIdx;
        if (copyLen > queElemLen) {
            copyLen = queElemLen;
        } else if (copyLen < 0) {
            copyLen = 0;
        }

        // 拉取本rank数据
        if (flagValue < sliceIdx) {
            sync.WaitInnerFlag(
                magic, sliceIdx, targetRank[idx], rank * coreNumPerRank + groupCoreIdx[idx] % coreNumPerRank);
            flagValue = sync.GetInnerFlag(
                targetRank[idx], rank * coreNumPerRank + groupCoreIdx[idx] % coreNumPerRank) & EVENT_ID_MASK;
        }
        readGt = readQue[idx].ReadFront();
        if (copyLen > 0) {
            LocalTensor<T> buffer1 = tempBuffer.AllocTensor<T>();
            LocalTensor<T> buffer2 = tempBuffer.AllocTensor<T>();
            int64_t remain = copyLen * sizeof(T);
            int64_t offset = 0;
            int64_t outOffset = 0;
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            int loop = 0;
            SetAtomicDataType<T>();
#ifdef __DAV_C220_VEC__
            SetAtomicOpType(0);
#endif
            while (remain > 0) {
                event_t eventId = (loop & 1) ? EVENT_ID0 : EVENT_ID1;
                wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
                // emb数量，每次最多拷贝半块UB大小的emb
                int64_t totalNum = remain < UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE ?
                    remain / dim / sizeof(T) : UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE / dim / sizeof(T);
                __ubuf__ T * buffer = (loop & 1) ?
                    (__ubuf__ T *)buffer1.GetPhyAddr() : (__ubuf__ T *)buffer2.GetPhyAddr();
                CpGM2UB(buffer, (__gm__ T *)readGt[offset].GetPhyAddr(), totalNum * dim * sizeof(T));
                offset += totalNum * dim;
                set_flag(PIPE_MTE2, PIPE_MTE3, eventId);
                wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
                for (int i = 0; i < totalNum; i++) {
                    int64_t outIdx = *(
                        (__gm__ int32_t *)restorePtr + sliceIdx * queElemLen / dim +
                        outputOffset[idx] / dim + outOffset + i
                    );
                    CpUB2GM(((__gm__ T*)outputPtr + outIdx * dim), buffer + i * dim, dim * sizeof(T));
                }
                set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
                remain -= UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE;
                outOffset += totalNum;
                loop += 1;
            }
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID3); // Scalar等MTE3
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID3);
            set_atomic_none();
            tempBuffer.FreeTensor(buffer1);
            tempBuffer.FreeTensor(buffer2);
        }
        sync.SetInnerFlag(magic, sliceIdx, rank, groupCoreIdx[idx] + flagNumPerStage);

        if (sliceIdx == sliceNum[idx] - 1) {
            sync.SetInnerFlag(1, 0, rank, groupCoreIdx[idx] + flagNumPerStage);
            sync.SetInnerFlag(1, 0, targetRank[idx], rank * coreNumPerRank + groupCoreIdx[idx] % coreNumPerRank);
        }
    }

    GlobalTensor <T> inputGt;
    GM_ADDR outputPtr;
    GlobalTensor <T> outputGt;
    GlobalTensor <T> outputGtInit;
    GlobalTensor <T> readGt;
    GlobalTensor <T> writeGt;
    GlobalTensor <int64_t> sendCountMatrixGm;
    GM_ADDR restorePtr;
    int64_t coreNumsPerStage;
    int64_t revLen = 0;
    int64_t sendLen = 0;
    int64_t sendOffset[MULTI_RANK_SIZE];
    int64_t revOffset[MULTI_RANK_SIZE];
    int64_t inputDataLen[MULTI_RANK_SIZE];
    int64_t outputDataLen[MULTI_RANK_SIZE];

    int waitRankListForWrite[MULTI_RANK_SIZE][1];  // 写共享内存时，需要等待的rank列表
    int waitNumForWrite[MULTI_RANK_SIZE];  // 写共享内存时，需要等待的数量
    int waitBlockForWrite[MULTI_RANK_SIZE];  // 写共享内存时，需要等待的标志位

    int64_t maxSliceNum;
    int64_t dim;
    int64_t queLen;
    int64_t queSize;
    int64_t coreNumPerStage;  // 每个阶段使用的核数
    int64_t flagNumPerStage;  // 每个阶段使用的同步标志位数
    int64_t coreNumPerRank;  // 每个rank数据分配的核数
    int64_t rankNumPerCore;  // 每个核负责的rank数
    int64_t coreGroup;  // 当前核的功能分组
    int64_t groupCoreIdx[MULTI_RANK_SIZE];  // 当前核在组内的索引，可以为等效核索引
    int64_t targetRank[MULTI_RANK_SIZE];  // 当前核负责的rank

    IpcQueue<T> readQue[MULTI_RANK_SIZE];  // 读端共享内存队列
    IpcQueue<T> writeQue[MULTI_RANK_SIZE];  // 写端共享内存队列
    TQue<QuePosition::VECIN, PING_PONG_SIZE> tempBuffer; // UB
    int64_t queElemLen;  // 共享内存队列里每个元素大小（以T计）

    int64_t sliceNum[MULTI_RANK_SIZE];  // 当前核负责的数据切片总数
    int64_t copyLen;  // 当前拷贝数据片的长度（以T计）
    int64_t inputOffset[MULTI_RANK_SIZE];  // 当前核负责的input偏移（以T计）
    int64_t inputLen[MULTI_RANK_SIZE];  // 当前核负责的input长度（以T计）
    int64_t outputOffset[MULTI_RANK_SIZE];  // 当前核负责的output偏移（以T计）
    int64_t outputLen[MULTI_RANK_SIZE];  // 当前核负责的output长度（以T计）
};

#endif // LCCL_ALL_USS_DETERMINISTIC_H
