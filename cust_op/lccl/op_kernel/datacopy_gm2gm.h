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

#ifndef LCCL_DATACOPY_GM2GM_H
#define LCCL_DATACOPY_GM2GM_H

#include "comm_args.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TILE_NUM = 2;
constexpr int32_t UB_OFFSET_START = 64;
constexpr int32_t BLOCK_SIZE = UB_SINGLE_DMA_SIZE_MAX / TILE_NUM / BUFFER_NUM;


template <typename T>
FORCE_INLINE_AICORE void SetAtomicDataType()
{
    if (std::is_same<T, float>::value) {
        set_atomic_f32();
    } else if (std::is_same<T, float16_t>::value) {
        set_atomic_f16();
    }
#ifdef __DAV_C220_VEC__
    else if (std::is_same<T, int>::value || std::is_same<T, int32_t>::value) {
        set_atomic_s32();
    } else if (std::is_same<T, int8_t>::value) {
        set_atomic_s8();
    } else if (std::is_same<T, int16_t>::value) {
        set_atomic_s16();
    } else if(std::is_same<T, bfloat16_t>::value) {
        set_atomic_bf16();
    } else {
        set_atomic_s32();
    }
#endif
}

FORCE_INLINE_AICORE void SetAtomicOpType(int op)
{
    switch (op) {
        case ADD:
            set_atomic_add();
            break;
        case MUL:
            // 累乘时忽略不设置atomic寄存器
            break;
        case MAX:
            set_atomic_max();
            break;
        case MIN:
            set_atomic_min();
            break;
        default:
            ;
    }
}

template <typename T>
class DataCopyGM2GM {
public:
    FORCE_INLINE_AICORE DataCopyGM2GM() {}

    FORCE_INLINE_AICORE void Init(const GlobalTensor<T>& outputGt, const GlobalTensor<T>& inputGt, const uint32_t calCount, int op)
    {
        inputGm = inputGt.GetPhyAddr();
        outputGm = outputGt.GetPhyAddr();
        inputUB = (__ubuf__ T*)get_imm(UB_OFFSET_START);
        this->op = op;
        dataSizeRemain = calCount * sizeof(T);
    }

    FORCE_INLINE_AICORE void Process()
    {
        int64_t i = 0;
        while (dataSizeRemain >= BLOCK_SIZE) {
            CpGM2UB(inputUB, (__gm__ T*)inputGm + i * BLOCK_SIZE / sizeof(T), BLOCK_SIZE);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);  // 3等2
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            CpUB2GM((__gm__ T*)outputGm + i * BLOCK_SIZE / sizeof(T), inputUB, BLOCK_SIZE);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);  // 2等3
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            i += 1;
            dataSizeRemain -= BLOCK_SIZE;
        }
        if (dataSizeRemain > 0) {
            CpGM2UB(inputUB, (__gm__ T*)inputGm + i * BLOCK_SIZE / sizeof(T), dataSizeRemain);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);  // 3等2
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            CpUB2GM((__gm__ T*)outputGm + i * BLOCK_SIZE / sizeof(T), inputUB, dataSizeRemain);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
    {
        pipe_barrier(PIPE_ALL);
        if(op != -1) {
            SetAtomicDataType<T>();
#ifdef __DAV_C220_VEC__
            SetAtomicOpType(op);
#endif
        }
        pipe_barrier(PIPE_ALL);

        copy_ubuf_to_gm_align_b8(gmAddr, ubAddr, 0, 1, size, 0, 0, 0, 0);
        if(op != -1) {
            set_atomic_none();
        }
    }

    FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
    {
        copy_gm_to_ubuf_align_b8(ubAddr, gmAddr, 0, 1, size, 0, 0, 0, 0);
    }

private:
    int64_t dataSizeRemain = 0;
    __ubuf__ T* inputUB;
    const __gm__ T* inputGm;
    const __gm__ T* outputGm;
    int op;
};

#endif //LCCL_DATACOPY_GM2GM_H