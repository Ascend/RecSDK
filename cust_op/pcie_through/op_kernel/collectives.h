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

#ifndef LCCL_COLLECTIVES_H
#define LCCL_COLLECTIVES_H

#include "comm_args.h"
using namespace AscendC;

class Collectives {
public:
    __aicore__ inline Collectives() {};
    __aicore__ inline void Init()
    {
        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();
    };

    __attribute__((always_inline)) inline __aicore__ void gm2ub(__ubuf__ uint8_t* ub_addr, __gm__ uint8_t* gm_addr,
    size_t data_size)
    {
        copy_gm_to_ubuf_align_b16(ub_addr, gm_addr, 0, 1, data_size, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    }

    __attribute__((always_inline)) inline __aicore__ void ub2gm(__gm__ uint8_t* gm_addr, __ubuf__ uint8_t* ub_addr,
    size_t data_size)
    {
        copy_ubuf_to_gm_align_b16(gm_addr, ub_addr, 0, 1, data_size, 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    template <typename T>
    __attribute__((always_inline)) inline __aicore__ void CpUB2GM(__gm__ T* gmAddr, __ubuf__ T* ubAddr, uint32_t size)
    {
#ifndef __GET_CODE_CHANNEL__
        pipe_barrier(PIPE_ALL);
        copy_ubuf_to_gm_align_b8(gmAddr, ubAddr, 0, 1, size, 0, 0, 0, 0);
        pipe_barrier(PIPE_ALL);
#endif
    }

    template <typename T>
    __attribute__((always_inline)) inline __aicore__ void CpGM2UB(__ubuf__ T* ubAddr, __gm__ T* gmAddr, uint32_t size)
    {
#ifndef __GET_CODE_CHANNEL__
        pipe_barrier(PIPE_ALL);
        copy_gm_to_ubuf_align_b8(ubAddr, gmAddr, 0, 1, size, 0, 0, 0, 0);
        pipe_barrier(PIPE_ALL);
#endif
    }
    //设置标志位
    __attribute__((always_inline)) inline __aicore__ void SetFlag(__ubuf__ uint64_t* ctrlFlagsUB,
                                                                  __gm__ uint64_t* ctrlFlagGM, uint64_t checkValue)
    {
        *ctrlFlagsUB = checkValue;
        CpUB2GM((__gm__ uint8_t* )ctrlFlagGM, (__ubuf__ uint8_t *)ctrlFlagsUB, sizeof(uint64_t));
    }
    //检查标志位
    __attribute__((always_inline)) inline __aicore__ void CheckFlag(__ubuf__ uint64_t* ctrlFlagsUB,
    __gm__ uint64_t* ctrlFlagGM, uint64_t checkValue)
    {
        while (true) {
            CpGM2UB((__ubuf__ uint8_t *)ctrlFlagsUB, (__gm__ uint8_t *)ctrlFlagGM, sizeof(uint64_t));
            if (*ctrlFlagsUB == checkValue) {
                break;
            }
        }
    }
    //获取标志位
    template <typename T>
    __attribute__((always_inline)) inline __aicore__ T GetFlag(__ubuf__ T* ctrlFlagsUB, __gm__ T* ctrlFlagGM)
    {
        CpGM2UB((__ubuf__ uint8_t *)ctrlFlagsUB, (__gm__ uint8_t *)ctrlFlagGM, sizeof(T));
        return *ctrlFlagsUB;
    }

    __attribute__((always_inline)) inline __aicore__ uint64_t GetFlag2(__ubuf__ uint64_t* ctrlFlagsUB,
                                                                       __gm__ uint64_t* ctrlFlagGM)
    {
        CpGM2UB((__ubuf__ uint8_t *)ctrlFlagsUB, (__gm__ uint8_t *)ctrlFlagGM, sizeof(uint64_t));
        return *ctrlFlagsUB;
    }
    //获得最小的标志位
    __attribute__((always_inline)) inline __aicore__ uint64_t GetMinFlag(__ubuf__ uint64_t* ctrlFlagsUB,
        __gm__ uint64_t** ctrlFlagGMs, int32_t num)
    {
        uint64_t minFlag = LLONG_MAX;
        for (int i = 0; i < num; ++i) {
            uint64_t val = GetFlag2(ctrlFlagsUB, ctrlFlagGMs[i]);
            if (minFlag > val) {
                minFlag = val;
            }
        }
        return minFlag;
    }
    //将数据从HBM的一块地址拷贝到另外一块地址
    __attribute__((always_inline)) inline __aicore__ void gm2gm(uint64_t data_size, __ubuf__ uint8_t* ub_buff,
    GM_ADDR dest_buff, GM_ADDR src_buff)
    {
        int32_t step = 0;
        while (data_size >= UNIT_COPY_SIZE) {
            gm2ub(ub_buff, src_buff + UNIT_COPY_SIZE * step, UNIT_COPY_SIZE);
            ub2gm(dest_buff + UNIT_COPY_SIZE * step, ub_buff, UNIT_COPY_SIZE);
            step += 1;
            data_size -= UNIT_COPY_SIZE;
        }
        if (data_size <= 0) {
            return;
        }
        gm2ub(ub_buff, src_buff + UNIT_COPY_SIZE * step, data_size);
        ub2gm(dest_buff + UNIT_COPY_SIZE * step, ub_buff, data_size);
    }

protected:
    int64_t blockIdx;
    int64_t blockNum;
};

#endif // LCCL_COLLECTIVES_H
