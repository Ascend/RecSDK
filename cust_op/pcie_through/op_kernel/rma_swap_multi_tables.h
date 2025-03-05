/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
constexpr uint64_t SPLIT_NUM = 2;
constexpr uint64_t LOOP_CHECK = 8;
constexpr uint64_t OUTFEED_CHECK = 4;

/**
 * @brief swap in & out operators
 * @tparam svmBuffSwapIn swap in queue
 * @tparam svmBuffSwapOut swap out queue
 */
class RmaSwapMultiTables : public Collectives {
public:
    __aicore__ inline RmaSwapMultiTables() : Collectives() {};

    __aicore__ inline void Init(GM_ADDR table_a, GM_ADDR table_b, GM_ADDR table_c, GM_ADDR table_d, GM_ADDR table_e,
                                GM_ADDR table_f, int tableNum, int tableLength, GM_ADDR swapInIndex,
                                GM_ADDR swapOutIndex, uint64_t swapInLen, GM_ADDR svmBuffSwapIn,
                                GM_ADDR svmBuffSwapOut, GM_ADDR usrWorkspace, int32_t dimNum, uint64_t *dimValue,
                                GM_ADDR output)
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        Collectives::Init();

        this->tableNum = tableNum;
        this->updateTables[0] = table_a; //第0个表
        this->updateTables[1] = table_b; //第1个表
        this->updateTables[2] = table_c; //第2个表
        this->updateTables[3] = table_d; //第3个表
        this->updateTables[4] = table_e; //第4个表
        this->updateTables[5] = table_f; //第5个表
        this->swapInIndex = swapInIndex;
        this->swapOutIndex = swapOutIndex;
        this->swapInLen = swapInLen;
        this->swapOutLen = dimValue[0];
        this->svmBuffSwapIn = svmBuffSwapIn;
        this->svmBuffSwapOut = svmBuffSwapOut;
        this->usrWorkspace = usrWorkspace;
        this->output = output;

        dataHeadSwapOut.dataType = 0;
        dataHeadSwapOut.dimNum = dimNum;
        for (int i = 0; i < dimNum - 1; ++i) {
            dataHeadSwapOut.dims[i] = dimValue[i]; // only support (length, emb_dim) with float32 type currently
        }
        dataHeadSwapOut.dims[dimNum - 1] = dimValue[dimNum - 1] * tableNum;
        embDim = dataHeadSwapOut.dims[1] * sizeof(float);      // host embedding dim(B)
        embDimSplit = dimValue[dimNum - 1] * sizeof(float);    // each table's emb dim = embDim // tableNum
        uint64_t totalLength = swapOutLen * embDim;
        dataHeadSwapOut.totalLen = totalLength + RMA_SHM_DATA_HEAD;
        dataHeadSwapOut.dataLen = totalLength;
        dataHeadSwapOut.readyLen = 0;
        swapFlagSwapIn = usrWorkspace + SWAP_IN_FLAG_OFFSET;
        swapFlagSwapOut = usrWorkspace + SWAP_OUT_FLAG_OFFSET;

        processBlockNum = blockNum / SPLIT_NUM;
        processBlockIdx = blockIdx % processBlockNum;
        GetQueHead();
        cacheCapacity = SWAP_CACHE_SIZE / embDim;
        cacheFront = 0;
        cacheRear = 0;
        SyncPreprocess();
    }

    __aicore__ inline void Process()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        if (GetFlag<uint64_t>(ub_buff, (__gm__ uint64_t *)output + blockIdx) != 0) {
            SyncPostprocess();
            return;
        }
        if (blockIdx < processBlockNum) {   // swap out
            if (processBlockIdx == 0) {
                OutfeedEnqueue();
            } else {
                LookUpTable();
            }
        } else {                            // swap in
            if (processBlockIdx < GET_NEXT_THREAD_NUM) {
                GetNextMultiThreads();
            } else {
                UpdateTable();
            }
        }
        SyncPostprocess();
    }

private:
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
        SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, 0);
        uint64_t times = 0;
        if (blockIdx < processBlockNum) {
            for (; times <= TIME_OUT; ++times) {
                ReadHeader(svmBuffSwapOut);
                if (!Full(dataHeadSwapOut.dataLen)) {
                    dataHeadSwapOut.sequence = queueHeader.seqIn + 1;
                    embSwapCache = usrWorkspace + SWAP_OUT_CACHE_OFFSET;
                    return;
                }
            }
            SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, RMA_QUEUE_TIME_OUT);
            return;
        } else {
            for (; times <= TIME_OUT; ++times) {
                ReadHeader(svmBuffSwapIn);
                if ((queueHeader.seqIn - queueHeader.seqOut) > 0) {
                    embSwapCache = usrWorkspace + SWAP_IN_CACHE_OFFSET;
                    return;
                }
            }
            SetFlag(ub_buff, (__gm__ uint64_t *)output + blockIdx, RMA_QUEUE_TIME_OUT);
            return;
        }
    }

    __aicore__ inline void LookUpTable()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)swapFlagSwapOut;
        __gm__ uint64_t *lookup_flag = (__gm__ uint64_t *)swapFlagSwapOut + processBlockIdx * FLAG_UNIT_INT_NUM;
        uint64_t outfeedCount = 0;
        uint64_t visitedIdx = processBlockIdx - 1;
        const uint64_t stride = processBlockNum - 1;
        cacheRear = visitedIdx % cacheCapacity;
        uint64_t loopCount = 0;
        while (visitedIdx < swapOutLen) {
            if (visitedIdx + 1 - outfeedCount >= cacheCapacity - 1) {    // cache is full
                outfeedCount = GetFlag2(ub_buff, outfeed_count);
                continue;
            }
            uint64_t embIdx = *((__gm__ uint64_t *)swapOutIndex + visitedIdx);
            for (int t = 0; t < tableNum; ++t) {
                gm2gm(embDimSplit, ub_data_buff, embSwapCache + cacheRear * embDim + t * embDimSplit,
                      updateTables[t] + embIdx * embDimSplit);
            }
            cacheRear = (cacheRear + stride) % cacheCapacity;
            visitedIdx += stride;
            if (loopCount % LOOP_CHECK == 0 || visitedIdx + 1 - outfeedCount >= cacheCapacity - 1) {
                SetFlag(ub_buff, lookup_flag, visitedIdx);
            }
            loopCount++;
        }
        SetFlag(ub_buff, lookup_flag, visitedIdx);
    }

    __aicore__ inline void OutfeedEnqueue()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *outfeed_count = (__gm__ uint64_t *)swapFlagSwapOut;
        __gm__ uint64_t *seqInSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_SEQ_IN_OFFSET;
        __gm__ uint64_t *tailSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_QUEUE_TAIL_OFFSET;
        __gm__ uint64_t *buffLimitSwapOut = (__gm__ uint64_t *)svmBuffSwapOut + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;
        GM_ADDR svmDataBuff;
        __gm__ uint64_t *svmReadyCount;

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
        if (queueHeader.tailOffset + ub_datahead_buff->totalLen > queueHeader.totalMemSize) {
            *ub_buff = queueHeader.tailOffset;
            CpUB2GM<uint64_t>(buffLimitSwapOut, ub_buff, sizeof(uint64_t));

            // write data head to swap out queue
            ub2gm(svmBuffSwapOut + RMA_SHM_HEAD_LEN, (__ubuf__ uint8_t *)ub_datahead_buff, RMA_SHM_DATA_HEAD);

            svmDataBuff = svmBuffSwapOut + RMA_SHM_HEAD_LEN + RMA_SHM_DATA_HEAD;
            svmReadyCount = (__gm__ uint64_t *)(svmBuffSwapOut + RMA_SHM_HEAD_LEN) + RMA_READY_LEN_OFFSET;
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
        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }
        pipe_barrier(PIPE_ALL);
        uint64_t lookUpCount = 0;
        uint64_t swapOutCount = 0;
        uint64_t loopCount = 0;
        while (swapOutCount < swapOutLen) {
            if (lookUpCount <= swapOutCount) {  // cache is empty
                lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                cacheRear = lookUpCount % cacheCapacity;
                continue;
            }
            uint64_t copyCount = (cacheCapacity + cacheRear - cacheFront) % cacheCapacity;
            if (cacheFront > cacheRear) {  // crocess tail of cache, address is discontinuity
                copyCount = cacheCapacity - cacheFront;
            }

            gm2gm(copyCount * dataHeadSwapOut.dims[1] * sizeof(float), ub_data_buff,
                  svmDataBuff + swapOutCount * embDim, embSwapCache + cacheFront * embDim);
            cacheFront = (cacheFront + copyCount) % cacheCapacity;
            swapOutCount += copyCount;
            if (loopCount % OUTFEED_CHECK == 0 || lookUpCount <= swapOutCount) {
                SetFlag(ub_buff, outfeed_count, swapOutCount);
            }
            loopCount++;
        }
        SetFlag(ub_buff, outfeed_count, swapOutCount);

        // update seqIn
        *ub_buff = dataHeadSwapOut.sequence;
        CpUB2GM<uint64_t>(seqInSwapOut, ub_buff, sizeof(uint64_t));
    }

    __aicore__ inline void UpdateTable()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(RMA_UB_B8_BUFF_OFFSET);
        __ubuf__ uint8_t *ub_data_buff = (__ubuf__ uint8_t *)get_imm(RMA_UB_DATA_BUFF_OFFSET);
        __gm__ uint64_t *update_flag = (__gm__ uint64_t *)swapFlagSwapIn + processBlockIdx * FLAG_UNIT_INT_NUM;
        __gm__ uint64_t *lookUpFlags[MAX_BLOCK_NUM];  // flags of swap out lookup table
        for (int i = 1; i < processBlockNum; ++i) {
            lookUpFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapOut + i * FLAG_UNIT_INT_NUM;
        }
        __gm__ uint64_t *getnextFlags[MAX_BLOCK_NUM];  // flags of swap in getnext
        for (int i = 0; i < GET_NEXT_THREAD_NUM; ++i) {
            getnextFlags[i] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }
        uint64_t getnextCount = 0;  // emb count has read from swap in queue
        const uint64_t freeCount = swapInLen - swapOutLen;  // free/invalide emb num in table
        uint64_t lookUpCount = 0;   // emb num has read from table
        uint64_t visitedIdx = processBlockIdx - GET_NEXT_THREAD_NUM;  // emb index to update
        const uint64_t stride = processBlockNum - GET_NEXT_THREAD_NUM;
        cacheFront = visitedIdx % cacheCapacity;
        uint64_t loopCount = 0;
        while (visitedIdx < swapInLen) {
            if (getnextCount <= visitedIdx || visitedIdx >= freeCount + lookUpCount) {
                if (getnextCount < swapInLen) {    // cache is empty
                    getnextCount = GetMinFlag(ub_buff, getnextFlags, GET_NEXT_THREAD_NUM);
                }
                if (lookUpCount < swapOutLen) {    // emb has not been read out from table
                    lookUpCount = GetMinFlag(ub_buff, lookUpFlags, processBlockNum - 1);
                }
                continue;
            }
            uint64_t embIdx = *((__gm__ uint64_t *)swapInIndex + visitedIdx);
            for (int t = 0; t < tableNum; ++t) {
                gm2gm(embDimSplit, ub_data_buff, updateTables[t] + embIdx * embDimSplit,
                      embSwapCache + cacheFront * embDim + t * embDimSplit);
            }
            visitedIdx += stride;
            cacheFront = visitedIdx % cacheCapacity;
            if (loopCount % LOOP_CHECK == 0 || getnextCount <= visitedIdx) {
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
        __gm__ uint64_t *seqOutSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_SEQ_OUT_OFFSET;
        __gm__ uint64_t *frontSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_QUEUE_FRONT_OFFSET;
        __gm__ uint64_t *buffLimitSwapIn = (__gm__ uint64_t *)svmBuffSwapIn + RmaQueueOffset::RMA_BUFF_LIMIT_OFFSET;

        bool updataBuffLimit = false;
        uint64_t frontOffset = queueHeader.frontOffset;
        if (queueHeader.buffLimit == frontOffset) {
            frontOffset = RMA_SHM_HEAD_LEN;
            updataBuffLimit = true;
        }
        CpGM2UB<uint8_t>((__ubuf__ uint8_t *)ub_datahead_buff, svmBuffSwapIn + frontOffset, RMA_SHM_DATA_HEAD);
        const uint64_t sizeOfTotalData = ub_datahead_buff->totalLen;
        const uint64_t sizeOfData = ub_datahead_buff->dataLen;
        const uint64_t sequence = ub_datahead_buff->sequence;
        GM_ADDR svmDataBuff = svmBuffSwapIn + frontOffset + RMA_SHM_DATA_HEAD;
        __gm__ uint64_t *ready_len = (__gm__ uint64_t *)(svmBuffSwapIn + frontOffset) + RMA_READY_LEN_OFFSET;
        pipe_barrier(PIPE_ALL);
        __gm__ uint64_t *updateFlags[MAX_BLOCK_NUM];
        for (int i = GET_NEXT_THREAD_NUM; i < processBlockNum; ++i) {
            updateFlags[i - GET_NEXT_THREAD_NUM] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
        }
        const uint64_t pipeBlockSize = ((2 * UNIT_COPY_SIZE) / embDim) * embDim;
        const uint64_t stride = pipeBlockSize * GET_NEXT_THREAD_NUM;

        if (sizeOfData > 0) {
            uint64_t updateCount = 0;   // emb count
            uint64_t readyLen = 0;      // Byte
            uint64_t copyOffset = processBlockIdx * pipeBlockSize;  // Byte
            uint64_t getnextCount = copyOffset / embDim;            // emb count
            cacheRear = (copyOffset / embDim) % cacheCapacity;
            while (copyOffset < sizeOfData) {
                if ((readyLen < sizeOfData && readyLen < copyOffset + pipeBlockSize) ||
                            (getnextCount >= updateCount && getnextCount - updateCount > cacheCapacity)) {
                    if (readyLen < sizeOfData) {
                        readyLen = GetFlag2(ub_buff, ready_len);
                    }
                    updateCount = GetMinFlag(ub_buff, updateFlags, processBlockNum - GET_NEXT_THREAD_NUM);
                    cacheFront = updateCount % cacheCapacity;
                    continue;
                }
                uint64_t copySize = (copyOffset + pipeBlockSize <= sizeOfData) ?
                                                                pipeBlockSize : (sizeOfData - copyOffset);
                uint64_t cacheSize = 0;
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
                gm2gm(copySize, ub_data_buff, embSwapCache + cacheRear * embDim, svmDataBuff + copyOffset);
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

        if (processBlockIdx == 0) {
            __gm__ uint64_t *getnextFlags[MAX_BLOCK_NUM];
            for (int i = 1; i < GET_NEXT_THREAD_NUM; ++i) {
                getnextFlags[i - 1] = (__gm__ uint64_t *)swapFlagSwapIn + i * FLAG_UNIT_INT_NUM;
            }
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

    __aicore__ inline void ReadHeader(GM_ADDR svm_buff)
    {
        __ubuf__ RmaShmHeader *ub_buff = (__ubuf__ RmaShmHeader *)get_imm(0);
        CpGM2UB<RmaShmHeader>(ub_buff, (__gm__ RmaShmHeader *)svm_buff, sizeof(RmaShmHeader));
        queueHeader.queueCapacity = ub_buff->queueCapacity;
        queueHeader.totalMemSize = ub_buff->totalMemSize;
        queueHeader.seqIn = ub_buff->seqIn;
        queueHeader.seqOut = ub_buff->seqOut;
        queueHeader.frontOffset = ub_buff->frontOffset;
        queueHeader.tailOffset = ub_buff->tailOffset;
        queueHeader.buffLimit = ub_buff->buffLimit;

        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ClearFlag()
    {
        if (blockIdx == 0) {
            __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
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

    __aicore__ inline void SyncPreprocess()
    {
        __ubuf__ uint64_t *ub_buff = (__ubuf__ uint64_t *)get_imm(0);
        __gm__ uint64_t *syncAllFlag = (__gm__ uint64_t *)swapFlagSwapIn + MAX_BLOCK_NUM * FLAG_UNIT_INT_NUM;
        if (blockIdx == 0) {
            ClearFlag();
            SetFlag(ub_buff, syncAllFlag, RMA_PRE_SYNC);
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