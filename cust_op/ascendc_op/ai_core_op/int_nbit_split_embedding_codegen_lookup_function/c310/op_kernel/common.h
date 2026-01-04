/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef INT_NBIT_SPLIT_EMBEDDING_COMMON_H
#define INT_NBIT_SPLIT_EMBEDDING_COMMON_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

// 常量定义
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_TYPE_INT32 = 0;
constexpr int DATA_TYPE_INT64 = 1;

constexpr int USE_QUEUE_NUM = 2;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int ALIGN = 32;
constexpr int64_t MAX_INDICES_ONE_BLOCK = 1024;
constexpr uint8_t FP8_SIGN_MASK = 0x80;
constexpr uint8_t FP8_BODY_MASK = 0x7F;

// FP8反量化相关常量
constexpr uint32_t FP8_SIGN_SHIFT = 24U;           // FP8符号位的左移量（用于位打包）
constexpr uint32_t FP8_BODY_SHIFT_OFFSET = 16U;   // FP8 body shift计算中的偏移量
constexpr uint32_t FP8_EXPONENT_MAX = 254U;        // FP8 exponent的最大值（用于multiplier计算）
constexpr uint32_t FLOAT32_EXPONENT_BIAS = 23U;   // IEEE 754 float32的指数位偏移量


// PoolingMode枚举
enum class PoolingMode {
    SUM = 0,
    MEAN = 1,
    NONE = 2
};

enum class PlacementType : uint8_t {
    DEVICE = 0,
    MANAGED = 1,
    MANAGED_CACHING = 2,
    HOST = 3
};

// SparseType枚举
enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    BF16 = 5,
    FP8 = 6,
    INVALID = 7
};

// Args结构体
struct Args {
    GM_ADDR devWeights;
    GM_ADDR uvmWeights;
    GM_ADDR lxuCacheWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR weightsTys;
    GM_ADDR dOffsets;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR lxuCacheLocations;
    GM_ADDR offsetPerKey;      // 每张表在offsets中的起始位置
    GM_ADDR indiceWeights;
    GM_ADDR out;
    GM_ADDR tiling;
    GM_ADDR workspace;
};

// ========== 数据拷贝函数 ==========

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(lt, gt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(gt, lt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
    }
}

// ========== FP8反量化公共函数 ==========

// 初始化FP8参数
__aicore__ inline void InitFp8Params(
    int64_t fp8ExponentBits,
    int64_t fp8ExponentBias,
    uint32_t& fp8BodyShift,
    float& fp8Multiplier)
{
    fp8BodyShift = static_cast<uint32_t>(fp8ExponentBits + FP8_BODY_SHIFT_OFFSET);
    union {
        uint32_t u32;
        float f32;
    } multUnion;
    multUnion.u32 = static_cast<uint32_t>(FP8_EXPONENT_MAX - fp8ExponentBias) << FLOAT32_EXPONENT_BIAS;
    fp8Multiplier = multUnion.f32;
}

// FP8到FP32的位打包转换
template <typename MaskBuf, typename TmpByteBuf, typename SignUint32Buf>
__aicore__ inline void FP8U8ToFP32Bitpack(
    const LocalTensor<uint8_t>& src,
    LocalTensor<float>& dst,
    int64_t elementCount,
    uint32_t fp8BodyShift,
    float fp8Multiplier,
    MaskBuf& fp8MaskBuf,
    TmpByteBuf& fp8TmpByteBuf,
    SignUint32Buf& fp8SignUint32Buf)
{
    if (elementCount <= 0) {
        return;
    }
    LocalTensor<uint8_t> maskLt = fp8MaskBuf.template Get<uint8_t>();
    LocalTensor<uint8_t> tmpU8Lt = fp8TmpByteBuf.template Get<uint8_t>();
    LocalTensor<uint32_t> signBitsLt = fp8SignUint32Buf.template Get<uint32_t>();
    LocalTensor<uint32_t> dstBitsLt = dst.ReinterpretCast<uint32_t>();

    Duplicate<uint8_t>(maskLt, FP8_SIGN_MASK, elementCount);
    And(tmpU8Lt, src, maskLt, elementCount);
    Cast(signBitsLt, tmpU8Lt, RoundMode::CAST_NONE, elementCount);
    ShiftLeft(signBitsLt, signBitsLt, FP8_SIGN_SHIFT, elementCount);

    Duplicate<uint8_t>(maskLt, FP8_BODY_MASK, elementCount);
    And(tmpU8Lt, src, maskLt, elementCount);
    Cast(dstBitsLt, tmpU8Lt, RoundMode::CAST_NONE, elementCount);
    ShiftLeft(dstBitsLt, dstBitsLt, fp8BodyShift, elementCount);
    LocalTensor<float> dstFloatLt = dstBitsLt.ReinterpretCast<float>();
    Muls(dstFloatLt, dstFloatLt, fp8Multiplier, elementCount);
    Or(dstBitsLt, dstBitsLt, signBitsLt, elementCount);
}

#endif  // INT_NBIT_SPLIT_EMBEDDING_COMMON_H
