/*
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

#ifndef LCCL_RMA_SWAP_MULTI_TABLES_H
#define LCCL_RMA_SWAP_MULTI_TABLES_H

#include "collectives.h"

using namespace AscendC;

constexpr uint64_t MAX_TABLE_NUM = 6;
constexpr uint64_t GET_NEXT_THREAD_NUM = 4;

#define RMA_SWAP_MULTI_TABLE_ARGS_FUN() \
GM_ADDR table_a, GM_ADDR table_b, GM_ADDR table_c, GM_ADDR table_d, GM_ADDR table_e, GM_ADDR table_f, \
int tableNum, int tableLength, GM_ADDR swapInIndex, GM_ADDR swapOutIndex, uint64_t swapInLen, \
GM_ADDR svmBuffSwapIn, GM_ADDR svmBuffSwapOut, GM_ADDR usrWorkspace, int32_t dimNum, uint64_t *dimValue, GM_ADDR output

#define RMA_SWAP_MULTI_TABLE_ARGS_CALL() \
table_a, table_b, table_c, table_d, table_e, table_f, \
tableNum, tableLength, swapInIndex, swapOutIndex, swapInLen, \
svmBuffSwapIn, svmBuffSwapOut, usrWorkspace, dimNum, dimValue, output

/**
 * @brief swap in & out operators
 * @tparam svmBuffSwapIn swap in queue
 * @tparam svmBuffSwapOut swap out queue
 */
class RmaSwapMultiTables : Collectives {
public:
    __aicore__ inline RmaSwapMultiTables() : Collectives() {};

    __aicore__ inline void Init(RMA_SWAP_MULTI_TABLE_ARGS_FUN())
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        Collectives::Init();

        this->tableNum = tableNum;
        this->updateTables[0] = table_a;
        this->updateTables[1] = table_b;
        this->updateTables[2] = table_c;
        this->updateTables[3] = table_d;
        this->updateTables[4] = table_e;
        this->updateTables[5] = table_f;
        this->swapInIndex = swapInIndex;
        this->swapOutIndex = swapOutIndex;
        this->swapInLen = swapInLen;
        this->swapOutLen = dimValue[0];
        //共享内存
        this->svmBuffSwapIn = svmBuffSwapIn;
        this->svmBuffSwapOut = svmBuffSwapOut;
        this->usrWorkspace = usrWorkspace;
        this->output = output;

        //保存从device拿到的换出信息到dataHeadSwapOut
        dataHeadSwapOut.dataType = 0;
        dataHeadSwapOut.dimNum = dimNum;
        for (int i = 0; i < dimNum - 1; ++i) {
            dataHeadSwapOut.dims[i] = dimValue[i]; // only support (length, emb_dim) with float32 type currently
        }
        //emb长度（val+slot）
        dataHeadSwapOut.dims[dimNum - 1] = dimValue[dimNum - 1] * tableNum;
        //（val+slot）的长度 * 4
        embDim = dataHeadSwapOut.dims[1] * sizeof(float);      // host embedding dim(B)
        //（val）的长度 * 4
        embDimSplit = dimValue[dimNum - 1] * sizeof(float);    // each table's emb dim = embDim // tableNum
        //换出的key数量 * 一条emb（val+slot）的大小=换出的数据量总的大小
        uint64_t totalLength = swapOutLen * embDim;
        //换出的数据量总大小+数据元素的头长度
        dataHeadSwapOut.totalLen = totalLength + RMA_SHM_DATA_HEAD;
        dataHeadSwapOut.dataLen = totalLength;
        dataHeadSwapOut.readyLen = 0;
        //换入的flag
        swapFlagSwapIn = usrWorkspace + SWAP_IN_FLAG_OFFSET;
        //换出的flag
        swapFlagSwapOut = usrWorkspace + SWAP_OUT_FLAG_OFFSET;

        processBlockNum = blockNum / 2;
        processBlockIdx = blockIdx % processBlockNum;

        GetQueHead();
        //缓存里面最多能放多少个key的emb（val+slot）
        cacheCapacity = SWAP_CACHE_SIZE / embDim;
        cacheFront = 0;
        cacheRear = 0;

        SyncPreprocess();
    }

    __aicore__ inline void Process()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        //先判断核的flag是不是0，如果不是，直接返回
        if (GetFlag<uint64_t>(ub_buff, (__gm__ uint64_t *)output + blockIdx) != 0) {
            SyncPostprocess();
            return;
        }
        //如果是换出的核
        if (blockIdx < processBlockNum) {   // swap out
            //0核用于搬运换出缓存的emb到共享内存
            if (processBlockIdx == 0) {
                OutfeedEnqueue();
            //23个核用于将表中需要换出的emb拷贝到换出缓存里面
            } else {
                LookUpTable();
            }
        //如果是换入的核
        } else {                            // swap in
            //0-3核用于
            if (processBlockIdx < GET_NEXT_THREAD_NUM) {
                GetNextMultiThreads();
            //其余的20个核
            } else {
                UpdateTable();
            }
        }
        SyncPostprocess();
    }
private:
    //判断队列是不是已经满了，满了返回true
    __aicore__ inline bool Full(uint64_t dataSize)
    {
        dataSize += RMA_SHM_DATA_HEAD;
        if (queueHeader.seqIn - queueHeader.seqOut >= queueHeader.queueCapacity) {
            return true;
        }
        if (queueHeader.tailOffset + dataSize > queueHeader.totalMemSize) {
            if (dataSize + RMA_SHM_HEAD_LEN > queueHeader.frontOffset) {
                return true;
            }
        } else {
            if (queueHeader.tailOffset < queueHeader.frontOffset &&
                        queueHeader.tailOffset + dataSize >= queueHeader.frontOffset) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void GetQueHead()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        //设置输出核的flag为0
        SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, 0);
        uint64_t times = 0;
        //换出队列，24个核
        if (blockIdx < processBlockNum) {
            //循环，直到队列有位置能放下换出的数据
            do {
                //获得队列头，保存到queueHeader
                ReadHeader(svmBuffSwapOut);
                //如果队列没有满，就停止循环
                if (!Full(dataHeadSwapOut.dataLen)) {
                    break;
                }
                //如果队列已经满了，就不断循环，直到超时，如果当前核超时，就设置当前核的flag为RMA_QUEUE_TIME_OUT
                if (++times > TIME_OUT) {
                    SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, RMA_QUEUE_TIME_OUT);
                    return;
                }
                // The queue is blocked when it is full.
            } while(true);
            //设置当前batch的序列号
            dataHeadSwapOut.sequence = queueHeader.seqIn + 1;
            //换出的缓存偏移地址
            embSwapCache = usrWorkspace + SWAP_OUT_CACHE_OFFSET;
        //换入队列，24个核
        } else {
            //循环，直到能从队列中获得H2D的数据
            do {
                ReadHeader(svmBuffSwapIn);
                //seqIn>seqOut,说明host向共享内存放的batch是多于device从共享内存获取的batch，说明队列中还是有未被device读取的数据的
                //那么device就可以继续往下处理了，
                if ((queueHeader.seqIn - queueHeader.seqOut) > 0) {
                    break;
                }
                if (++times > TIME_OUT) {
                    SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, RMA_QUEUE_TIME_OUT);
                    return;
                }
                // The queue is blocked when it is empty.
            } while (true);
            //换入的缓存偏移地址
            embSwapCache = usrWorkspace + SWAP_IN_CACHE_OFFSET;
        }
    }
    //23个核处理换出
    __aicore__ inline void LookUpTable()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)swapFlagSwapOut;
        __gm__ uint64_t *lookup_flag = (__gm__ uint64_t *)swapFlagSwapOut + processBlockIdx * FLAG_UNIT_INT_NUM;
        //outfeedCount维护0核
        uint64_t outfeedCount = 0;
        //visitedIdx维护23个核向共享内存中更新emb的最新索引
        uint64_t visitedIdx = processBlockIdx - 1; // 最开始每个核处理key的索引，
        const uint64_t stride = processBlockNum - 1;  //下面的循环，每次23个核并发处理key的emb
        cacheRear = visitedIdx % cacheCapacity;
        uint64_t loopCount = 0;
        while (visitedIdx < swapOutLen) {
            //判断23个核向换出缓存中写入的emb是不是已经堆满缓存了，如果堆满了，就循环获得outfeedCount，直到有缓存
            if (visitedIdx + 1 - outfeedCount >= cacheCapacity - 1) {    // cache is full
                outfeedCount = GetFlag2(ub_buff, outfeed_count);
                continue;
            }
            //swapOutIndex为换出的索引列表，swapOutIndex + visitedIdx为当前核处理的表的emb索引
            uint64_t embIdx = *((__gm__ uint64_t *)swapOutIndex + visitedIdx);
            for (int t = 0; t < tableNum; ++t) {
                //embDimSplit: (val）的长度 * 4
                //换出的缓存地址 +  （val+slot）的长度 * 4 + 当前表 * (val）的长度 * 4
                //updateTables[t] + embIdx * embDimSplit ：当前核处理的emb
                //拷贝表里面需要换出的emb到换出缓存里面
                gm2gm(embDimSplit, ub_data_buff, embSwapCache + cacheRear * embDim + t * embDimSplit,
                      updateTables[t] + embIdx * embDimSplit);
            }
            cacheRear = (cacheRear + stride) % cacheCapacity;
            visitedIdx += stride; //每个核下一轮要处理的key的索引，需要+23，因为一次循环里面有23个key被23个核一起处理
            //更新换出flag，每个核都有一个flag，更新flag为最新的visitedIdx
            if (loopCount % 8 == 0 || visitedIdx + 1 - outfeedCount >= cacheCapacity - 1) {
                SetFlag(ub_buff, lookup_flag, visitedIdx);
            }
            loopCount++;
        }
        SetFlag(ub_buff, lookup_flag, visitedIdx);
    }

    //第0个换出核处理
    __aicore__ inline void OutfeedEnqueue()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)swapFlagSwapOut;
        //RmaShmHeader。seqIn的地址
        __gm__ uint64_t *seqInSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_SEQ_IN_OFFSET;
        //RmaShmHeader。tailOffset的地址
        __gm__ uint64_t *tailSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_QUEUE_TAIL_OFFSET;
        //RmaShmHeader。buffLimit的地址
        __gm__ uint64_t *buffLimitSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;
        GM_ADDR svmDataBuff;
        __gm__ uint64_t *svmReadyCount;

        // generating data header information
        //更新换出的数据head到ub里面
        __ubuf__ RmaShmDataHead *ub_datahead_buff = (__ubuf__ RmaShmDataHead *)get_imm(0);
        ub_datahead_buff->totalLen = dataHeadSwapOut.totalLen;
        ub_datahead_buff->sequence = dataHeadSwapOut.sequence;
        ub_datahead_buff->dataType = dataHeadSwapOut.dataType;
        ub_datahead_buff->dimNum = dataHeadSwapOut.dimNum;
        for (int i = 0; i < dataHeadSwapOut.dimNum; ++i) {
            ub_datahead_buff->dims[i] = dataHeadSwapOut.dims[i] ;
        }
        ub_datahead_buff->dataLen = dataHeadSwapOut.dataLen;
        ub_datahead_buff->readyLen = dataHeadSwapOut.readyLen;
        pipe_barrier(PIPE_ALL);

        // free space in queue's tail is not enough, put data from queue's begin pos
        //更新数据head和队列 head到共享内存里面，也就是HBM里面
        //队列满了，重新从头开始写
        if (queueHeader.tailOffset + ub_datahead_buff->totalLen > queueHeader.totalMemSize) {
            //拷贝tailOffset到buffLimit，作为队列尾部无法写入数据的偏移
            *ub_buff = queueHeader.tailOffset;
            CpUB2GM<uint64_t>(buffLimitSwapOut, ub_buff, sizeof(uint64_t));

            // write data head to swap out queue
            //把换出的数据head从ub拷贝到共享内存里面，也就是HBM里面
            ub2gm(svmBuffSwapOut + RMA_SHM_HEAD_LEN, (__ubuf__ uint8_t *)ub_datahead_buff, RMA_SHM_DATA_HEAD);

            //换出队列head更新到共享内存里面，也就是HBM里面
            //换出的数据内容地址=共享内存地址+队列头长度+数据头长度
            svmDataBuff = svmBuffSwapOut + RMA_SHM_HEAD_LEN + RMA_SHM_DATA_HEAD;
            //RmaShmData，readyLen的地址
            svmReadyCount = (__gm__ uint64_t *)(svmBuffSwapOut + RMA_SHM_HEAD_LEN) + RMA_READY_LEN_OFFSET;
            //获得tailOffset的地址，可写入数据的地址偏移量，从UB拷贝到HBM里面的RmaShmHeader。tailOffset，也就是共享内存里面的
            *ub_buff = RMA_SHM_HEAD_LEN + ub_datahead_buff->totalLen;
            CpUB2GM<uint64_t>(tailSwapOut, ub_buff, sizeof(uint64_t));
        } else {
            // write data head to swap out queue
            ub2gm(svmBuffSwapOut + queueHeader.tailOffset, (__ubuf__ uint8_t *)ub_datahead_buff, RMA_SHM_DATA_HEAD);
            svmDataBuff = svmBuffSwapOut + queueHeader.tailOffset + RMA_SHM_DATA_HEAD;
            svmReadyCount = (__gm__ uint64_t *)(svmBuffSwapOut + queueHeader.tailOffset) + RMA_READY_LEN_OFFSET;
            *ub_buff = queueHeader.tailOffset + ub_datahead_buff->totalLen;
            CpUB2GM<uint64_t>(tailSwapOut, ub_buff, sizeof(uint64_t));
        }
        //获得换出的flag，每个核一个flag，一个24个，保存到lookUpFlags里面
        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }
        pipe_barrier(PIPE_ALL);
        uint64_t lookUpCount = 0;
        //已经从缓存拷贝到共享内存里面的总数据量
        uint64_t swapOutCount = 0;
        uint64_t loopCount = 0;
        while (swapOutCount < swapOutLen) {
            //lookUpCount记录了23个核向缓存中写入emb的最早的位置
            //swapOutCount记录了0核向共享内存中写入emb的最新位置
            if (lookUpCount <= swapOutCount) {  // cache is empty
                //获得23个核最小的flag，flag为最新的visitedIdx
                lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                //获得23个核向缓存中写emb的最早的位置
                cacheRear = lookUpCount % cacheCapacity;
                continue;
            }
            //23个核向缓存中写emb，会更新cacheRear
            //0核将emb从缓存向共享内存搬运，会更新cacheFront。
            //cacheRear - cacheFront 可以得到本次循环，0核需要向共享内存中搬运多少emb
            uint64_t copyCount = (cacheCapacity + cacheRear - cacheFront) % cacheCapacity;
            if (cacheFront > cacheRear) {  // crocess tail of cache, address is discontinuity
                copyCount = cacheCapacity - cacheFront;
            }
            //拷贝缓存里面的数据到共享内存里面
            //拷贝copyCount个emb
            gm2gm(copyCount * dataHeadSwapOut.dims[1] * sizeof(float), ub_data_buff,
                  svmDataBuff + swapOutCount * embDim, embSwapCache + cacheFront * embDim);
            //更新拷贝缓存到共享内存里面的最新地址
            cacheFront = (cacheFront + copyCount) % cacheCapacity;
            swapOutCount += copyCount;
            //每4次训练，就更新一次0核的flag为swapOutCount
            if (loopCount % 4 == 0 || lookUpCount <= swapOutCount) {
                SetFlag(ub_buff, outfeed_count, swapOutCount);
            }
            loopCount++;
        }
        SetFlag(ub_buff, outfeed_count, swapOutCount);

        // update seqIn
        //更新队列的RmaShmHeader。seqIn，
        *ub_buff = dataHeadSwapOut.sequence;
        CpUB2GM<uint64_t>(seqInSwapOut, ub_buff, sizeof(uint64_t));
    }
    //20个核用于将换入缓存里面的emb更新到表里面
    __aicore__ inline void UpdateTable()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *update_flag = (__gm__ uint64_t *)swapFlagSwapIn + processBlockIdx * FLAG_UNIT_INT_NUM;
        //换出的23个核的flag
        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];  // flags of swap out lookup table
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }
        //换入的4个核的flag
        __gm__ uint64_t *getnextFlags[MAX_BLOCK_NUM];  // flags of swap in getnext
        for (int i = 0; i < GET_NEXT_THREAD_NUM; ++i) {
            getnextFlags[i] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }
        //共享内存写入换入缓存的最早的索引
        uint64_t getnextCount = 0;  // emb count has read from swap in queue
        const uint64_t freeCount = swapInLen - swapOutLen;  // free/invalide emb num in table
        uint64_t lookUpCount = 0;   // emb num has read from table
        uint64_t visitedIdx = processBlockIdx - GET_NEXT_THREAD_NUM;  // emb index to update
        //20个核
        const uint64_t stride = processBlockNum - GET_NEXT_THREAD_NUM;
        cacheFront = visitedIdx % cacheCapacity;
        uint64_t loopCount = 0;
        while (visitedIdx < swapInLen) {
            if (getnextCount <= visitedIdx || visitedIdx >= freeCount + lookUpCount) {
                if (getnextCount < swapInLen) {    // cache is empty
                    //共享内存写入换入缓存的flag，4个核中的最小值
                    getnextCount = GetMinFlag(ub_buff, getnextFlags, GET_NEXT_THREAD_NUM);
                }
                //保证先换出再换入
                if (lookUpCount < swapOutLen) {    // emb has not been read out from table
                    lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                }
                continue;
            }
            //swapInIndex为换入的索引列表，swapInIndex + visitedIdx为当前核处理的表的emb索引
            uint64_t embIdx = *((__gm__ uint64_t *)swapInIndex + visitedIdx);
            //将换入缓存中的emb更新到table里面
            for (int t = 0; t < tableNum; ++t) {
                gm2gm(embDimSplit, ub_data_buff, updateTables[t] + embIdx * embDimSplit,
                      embSwapCache + cacheFront * embDim + t * embDimSplit);
            }
            //每次训练，20个核并行处理
            visitedIdx += stride;
            cacheFront = visitedIdx % cacheCapacity;
            //更新核的flag为最新的写入table的索引
            if (loopCount % 8 == 0 || getnextCount <= visitedIdx) {
                SetFlag(ub_buff, update_flag, visitedIdx);
            }
            loopCount++;
        }
        SetFlag(ub_buff, update_flag, visitedIdx);
    }

    __aicore__ inline void GetNextMultiThreads()
    {
        __ubuf__ RmaShmDataHead *ub_datahead_buff = (__ubuf__ RmaShmDataHead *)get_imm(0);
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *getnext_count = (__gm__ uint64_t *)swapFlagSwapIn + processBlockIdx * FLAG_UNIT_INT_NUM;
        //RmaShmHeader。seqIn的地址
        __gm__ uint64_t *seqOutSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_SEQ_OUT_OFFSET;
        //RmaShmHeader。frontOffset的地址
        __gm__ uint64_t *frontSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_QUEUE_FRONT_OFFSET;
        //RmaShmHeader。buffLimit的地址
        __gm__ uint64_t *buffLimitSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;

        //判断是否需要重头访问，如果需要，就设置frontOffset为队列head的位置
        bool updataBuffLimit = false;
        uint64_t frontOffset = queueHeader.frontOffset;
        if (queueHeader.buffLimit == frontOffset) {
            frontOffset = RMA_SHM_HEAD_LEN;
            updataBuffLimit = true;
        }
        //拷贝共享内存一个batch的数据头到UB里面
        CpGM2UB<uint8_t>((__ubuf__ uint8_t *)ub_datahead_buff, svmBuffSwapIn + frontOffset, RMA_SHM_DATA_HEAD);
        //获得数据头的信息
        const uint64_t sizeOfTotalData = ub_datahead_buff->totalLen;
        const uint64_t sizeOfData = ub_datahead_buff->dataLen;
        const uint64_t sequence = ub_datahead_buff->sequence;
        //获得共享内存里面这个batch的实际数据的开始地址
        GM_ADDR svmDataBuff = svmBuffSwapIn + frontOffset + RMA_SHM_DATA_HEAD;
        //获得这个batch已准备好的数据长度
        __gm__ uint64_t *ready_len = (__gm__ uint64_t *)(svmBuffSwapIn + frontOffset) + RMA_READY_LEN_OFFSET;
        pipe_barrier(PIPE_ALL);
        //获得20个核的flag
        __gm__ uint64_t *updateFlags[MAX_BLOCK_NUM];
        for (int i = GET_NEXT_THREAD_NUM; i < processBlockNum; ++i) {
            updateFlags[i - GET_NEXT_THREAD_NUM] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }
        //H2D的带宽比D2H高，所以gm2gm一次的数据量就可以多一些，每次拷贝的数据量就不是OutfeedEnqueue函数里面的一个emb
        //而且H2D是4个核，D2H是1个核
        const uint64_t pipeBlockSize = ((2 * UNIT_COPY_SIZE) / embDim) * embDim;
        const uint64_t stride = pipeBlockSize * GET_NEXT_THREAD_NUM;

        if (sizeOfData > 0) {
            uint64_t updateCount = 0;   // emb count
            uint64_t readyLen = 0;      // Byte
            //当前核拷贝的偏移地址
            uint64_t copyOffset = processBlockIdx * pipeBlockSize;  // Byte
            //当前核写入缓存的最新位置
            uint64_t getnextCount = copyOffset / embDim;            // emb count
            cacheRear = (copyOffset / embDim) % cacheCapacity;
            while (copyOffset < sizeOfData) {
                if ((readyLen < sizeOfData && readyLen < copyOffset + pipeBlockSize) ||
                            (getnextCount >= updateCount && getnextCount - updateCount > cacheCapacity)) {
                    //获得准备好的数据长度
                    if (readyLen < sizeOfData) {
                        readyLen = GetFlag2(ub_buff, ready_len);
                    }
                    //读取缓存的最新偏移位置
                    updateCount = GetMinFlag(ub_buff, updateFlags, processBlockNum - GET_NEXT_THREAD_NUM);
                    cacheFront = updateCount % cacheCapacity;
                    continue;
                }
                //拷贝的数据量大小
                uint64_t copySize = (copyOffset + pipeBlockSize <= sizeOfData) ?
                                                                pipeBlockSize : (sizeOfData - copyOffset);
                uint64_t cacheSize = 0;
                //cacheRear：写入缓存的最新偏移位置
                //cacheFront：读取缓存的最新偏移位置
                //？？？？？这个地方怎么计算的
                if (cacheRear >= cacheFront) {
                    if (cacheFront == 0) {
                        cacheSize = (cacheCapacity - cacheRear - 1) * embDim;
                    } else {
                        cacheSize = (cacheCapacity - cacheRear) * embDim;
                    }
                } else {
                    cacheSize = (cacheFront - cacheRear - 1) * embDim;
                }

                copySize = (copySize > cacheSize) ? cacheSize : copySize;
                //共享内存向缓存拷贝数据，每次拷贝copySize个
                gm2gm(copySize, ub_data_buff, embSwapCache + cacheRear * embDim, svmDataBuff + copyOffset);
                //更新偏移的位置
                if (copyOffset + stride >= sizeOfData) {
                    copyOffset = sizeOfData;
                } else {
                    copyOffset += stride;
                }
                getnextCount = copyOffset / embDim;
                cacheRear = getnextCount % cacheCapacity;
                SetFlag(ub_buff, getnext_count, getnextCount);
            }
            SetFlag(ub_buff, getnext_count, getnextCount);
        }
        //用0核来更新数据头的信息
        if (processBlockIdx == 0) {
            __gm__ uint64_t *getnextFlags[MAX_BLOCK_NUM];
            for (int i = 1; i < GET_NEXT_THREAD_NUM; ++i) {
                getnextFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
            }
            //保证所有的换入数据都已经从共享内存拷贝到缓存里面，才更新数据头信息
            uint64_t minGetnext = 0;
            while (minGetnext < swapInLen) {
                minGetnext = GetMinFlag(ub_buff, getnextFlags, GET_NEXT_THREAD_NUM - 1);
            }

            if (updataBuffLimit) {
                *ub_buff = 0;
                CpUB2GM<uint64_t>(buffLimitSwapIn, ub_buff, sizeof(uint64_t));
            }

            // update front offset
            *ub_buff = frontOffset + sizeOfTotalData;
            CpUB2GM<uint64_t>(frontSwapIn, ub_buff, sizeof(uint64_t));
            // update seqOut
            *ub_buff = sequence;
            CpUB2GM<uint64_t>(seqOutSwapIn, ub_buff, sizeof(uint64_t));
        }
    }
    //svm_buff为共享内存的地址，读取共享内存里面的队列头内容到queueHeader
    __aicore__ inline void ReadHeader(GM_ADDR svm_buff)
    {
        __ubuf__ RmaShmHeader *ub_buff = (__ubuf__ RmaShmHeader *)get_imm(0);
        //拷贝共享内存里面队列头到UB里面
        CpGM2UB<RmaShmHeader>(ub_buff, (__gm__ RmaShmHeader *)svm_buff, sizeof(RmaShmHeader));
        //队列深度
        queueHeader.queueCapacity = ub_buff->queueCapacity;
        //共享内存大小
        queueHeader.totalMemSize = ub_buff->totalMemSize;
        //最新写入的seq
        queueHeader.seqIn = ub_buff->seqIn;
        //最新读取的seq
        queueHeader.seqOut = ub_buff->seqOut;
        //可访问的内存地址偏移量
        queueHeader.frontOffset = ub_buff->frontOffset;
        //可写入数据的地址偏移量
        queueHeader.tailOffset = ub_buff->tailOffset;
        //队列尾部无法写入数据的偏移，标识队列需要返回到头部进行写入
        queueHeader.buffLimit = ub_buff->buffLimit;

        pipe_barrier(PIPE_ALL);
    }
    //设置核的flag为0
    __aicore__ inline void ClearFlag()
    {
        if (blockIdx == 0) {
            __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
            //这里为啥是48*2，flagNum不是一个核一个吗？
            const int flagNum = MAX_BLOCK_NUM * 2;
            for (int i = 0; i < flagNum * FLAG_UNIT_INT_NUM; ++i) {
                *(ub_buff + i) = 0;
            }
            CpUB2GM<uint8_t>(swapFlagSwapIn, (__ubuf__ uint8_t *)ub_buff,
                             flagNum * FLAG_UNIT_INT_NUM * sizeof(uint64_t));
            CpUB2GM<uint8_t>(swapFlagSwapOut, (__ubuf__ uint8_t *)ub_buff,
                             flagNum * FLAG_UNIT_INT_NUM * sizeof(uint64_t));
        }
    }
    //    |---48---|1|---47--|--------------------------|-----------------------------------------------------------|
    //      1Mflag缓存，只有前面的48*2个空间有用，其余都是无用                                  100M数据缓存
    //  48个核flag 1个同步flag 47个空flag
    __aicore__ inline void SyncPreprocess()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __gm__ uint64_t *syncAllFlag = (__gm__ uint64_t *)swapFlagSwapIn + MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM;
        //0核会进行初始化48*2个空间，并设置同步flag，
        if (blockIdx == 0) {
            //设置1M缓存里面的前48*2个空间为0
            ClearFlag();
            //设置同步flag为RMA_PRE_SYNC
            SetFlag(ub_buff, syncAllFlag, RMA_PRE_SYNC);
        //其他核会在CheckFlag里面不断的循环，直到 同步flag被0核设置为RMA_PRE_SYNC，这也意味着0核已经完成了48*2的空间初始化了
        } else {
            CheckFlag(ub_buff, syncAllFlag, RMA_PRE_SYNC);
        }
    }

    __aicore__ inline void SyncPostprocess()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __gm__ uint64_t *syncAllFlag = (__gm__ uint64_t *)swapFlagSwapOut + MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM;
        if (blockIdx != 0) {
            SetFlag(ub_buff, syncAllFlag + blockIdx * FLAG_UNIT_INT_NUM, RMA_POST_SYNC);
        } else {
            for (int i = 1; i < blockNum; ++i) {
                CheckFlag(ub_buff, syncAllFlag + i * FLAG_UNIT_INT_NUM, RMA_POST_SYNC);
            }
            ClearFlag();
        }
    }

private:
    //队列头
    RmaShmHeader queueHeader;
    RmaShmDataHead dataHeadSwapOut;
    GM_ADDR updateTables[MAX_TABLE_NUM];
    uint64_t tableNum;
    GM_ADDR swapInIndex;
    GM_ADDR swapOutIndex;
    uint64_t swapInLen;
    uint64_t swapOutLen;
    GM_ADDR svmBuffSwapIn;
    GM_ADDR svmBuffSwapOut;
    GM_ADDR usrWorkspace;
    GM_ADDR swapFlagSwapIn;
    GM_ADDR swapFlagSwapOut;
    uint64_t embDim;
    uint64_t embDimSplit;
    GM_ADDR embSwapCache;
    uint64_t cacheCapacity;
    uint64_t cacheFront;
    uint64_t cacheRear;
    GM_ADDR output;
    uint32_t processBlockNum;
    uint32_t processBlockIdx;
};

#endif // LCCL_RMA_SWAP_MULTI_TABLES_H