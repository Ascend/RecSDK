/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef GATHER_FOR_RANK1_FUN_H
#define GATHER_FOR_RANK1_FUN_H

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace GatherForRank1 {

constexpr int USE_QUEUE_NUM = 2;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_XDIM0 = 20480;
constexpr int DATA_COPY_PAD_ALIGN_BYTE2 = 16;
constexpr int DATA_COPY_PAD_ALIGN_BYTE4 = 8;
#ifdef SUPPORT_V200
    constexpr int ONEBLOCK_ELEM = 8192;
#else
    constexpr int ONEBLOCK_ELEM = 4096;
#endif
struct Args {
    GM_ADDR x;
    GM_ADDR index;
    GM_ADDR y;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename xType>
class GatherForRank1Kernel {
public:
    __aicore__ inline GatherForRank1Kernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        // ADDR
        x = args.x;
        index = args.index;
        y = args.y;

        // Shape
        xDim0 = tilingData.xDim0;
        indexDim0 = tilingData.indexDim0;

        // Ub
        ubCanUsed = tilingData.ubCanUsed;
        oneBlockElem = ONEBLOCK_ELEM;

        // Gt
        xGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x), xDim0);
        yGt.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(y), indexDim0);

        // Pipe
        pipe.InitBuffer(xInQue, 1, MAX_XDIM0 * sizeof(xType));
        pipe.InitBuffer(outQue, 1, ONEBLOCK_ELEM * sizeof(xType));

#ifdef SUPPORT_V200
        indexGt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), indexDim0);
        pipe.InitBuffer(indexInQue, 1, ONEBLOCK_ELEM * sizeof(int32_t));
#else
        indexGt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(index), indexDim0);
        pipe.InitBuffer(indexInQue, 1, ONEBLOCK_ELEM * sizeof(int64_t));
        pipe.InitBuffer(indexCastUint32Buf, ONEBLOCK_ELEM * sizeof(uint32_t));
#endif
    }

    template <typename T>
    __aicore__ inline void DataCopyPadGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T> &gt, int64_t len)
    {
        uint32_t dataCopyPadLen = DATA_COPY_PAD_ALIGN_BYTE2;
        TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        if constexpr (sizeof(T) == 4) {  // float、int32为4字节数据
            dataCopyPadLen = DATA_COPY_PAD_ALIGN_BYTE4;
            DataCopy<T>(lt, gt, dataCopyPadLen);
            SetFlag<HardEvent::MTE3_V>(eventId);
            WaitFlag<HardEvent::MTE3_V>(eventId);
            uint64_t mask0 = (1uL << dataCopyPadLen) - (1uL << len);
            uint64_t mask[1] = { mask0 };
            Duplicate<T>(lt, 0, mask, 1, 1, 1);
        } else {
            DataCopy<T>(lt, gt, dataCopyPadLen);
            SetFlag<HardEvent::MTE3_V>(eventId);
            WaitFlag<HardEvent::MTE3_V>(eventId);
            uint64_t mask0 = (1uL << dataCopyPadLen) - (1uL << len);
            uint64_t mask[2] = { mask0, 0 };
            Duplicate<T>(lt, 0, mask, 1, 1, 1);
        }
    }

    template <typename T>
    __aicore__ inline void CpPadGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        if (alignLen != 0) {
            DataCopy(lt, gt, alignLen / sizeof(T));
        }
        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
        DataCopyPadGm2Local(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], unAlignLen / sizeof(T));
#else
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
#endif
        }
    }

    template <typename T>
    __aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        GlobalTensor<int32_t> int32Gt;
        int32Gt.SetGlobalBuffer((__gm__ int32_t*)gt.GetPhyAddr(), len * sizeof(T) / sizeof(int32_t));
        LocalTensor<int32_t> int32Lt = lt.template ReinterpretCast<int32_t>();

        int64_t transLen = len * sizeof(T) / sizeof(int32_t);
        CpPadGm2Local(int32Lt, int32Gt, transLen);
    }

    __aicore__ inline void DataCopyPadLocal2Gm(const GlobalTensor<xType>& gt, const LocalTensor<xType>& lt,
        int64_t len)
    {
        SetAtomicAdd<xType>();
        uint32_t dataCopyPadLen = DATA_COPY_PAD_ALIGN_BYTE2;
        if constexpr (std::is_same<xType, float>::value) {
            dataCopyPadLen = DATA_COPY_PAD_ALIGN_BYTE4;
            uint64_t mask0 = (1uL << dataCopyPadLen) - (1uL << len);
            uint64_t mask[1] = { mask0 };
            Duplicate<xType>(lt, 0, mask, 1, 1, 1);
        } else {
            uint64_t mask0 = (1uL << dataCopyPadLen) - (1uL << len);
            uint64_t mask[2] = { mask0, 0 };
            Duplicate<xType>(lt, 0, mask, 1, 1, 1);
        }
        
        pipe_barrier(PIPE_ALL);
        TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        DataCopy<xType>(gt, lt, dataCopyPadLen);

        pipe_barrier(PIPE_ALL);
        SetAtomicNone();
    }

    __aicore__ inline void CpLocal2Gm(const GlobalTensor<xType>& gt, const LocalTensor<xType>& lt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(xType) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(xType) - alignLen;

        if (alignLen != 0) {
            DataCopy(gt, lt, alignLen / sizeof(xType));
        }
        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadLocal2Gm(gt[alignLen / sizeof(xType)], lt[alignLen / sizeof(xType)], unAlignLen / sizeof(xType));
#else
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            DataCopyPad(gt[alignLen / sizeof(xType)], lt[alignLen / sizeof(xType)], dataCopyExtParams);
#endif
        }
    }

    __aicore__ inline int64_t Ceil(int64_t a, int64_t b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b * b;
    }
    __aicore__ inline void GatherX(LocalTensor<xType>& xLt, int64_t offsetOfThisCore, int64_t lenOfThisCore)
    {
        int64_t total = lenOfThisCore;
        int64_t remain = total;
        int64_t offset = 0;
        while (remain > 0) {
            int64_t thisLen = oneBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t thisOffset = offsetOfThisCore + total - remain;
#ifdef SUPPORT_V200
            int64_t paddingLen = Ceil(thisLen, DATA_ALIGN_BYTES / sizeof(int32_t));
            LocalTensor<int32_t> indexInLt = indexInQue.AllocTensor<int32_t>();
            CpGm2Local(indexInLt, indexGt[thisOffset], thisLen);

            indexInQue.EnQue(indexInLt);
            indexInLt = indexInQue.DeQue<int32_t>();
            pipe_barrier(PIPE_ALL);
#else
            int64_t paddingLen = Ceil(thisLen, DATA_ALIGN_BYTES / sizeof(int64_t));

            LocalTensor<int64_t> indexIn64Lt = indexInQue.AllocTensor<int64_t>();
            CpGm2Local(indexIn64Lt, indexGt[thisOffset], thisLen);

            indexInQue.EnQue(indexIn64Lt);
            indexIn64Lt = indexInQue.DeQue<int64_t>();

            LocalTensor<int32_t> indexInLt = indexCastUint32Buf.Get<int32_t>();
            Cast(indexInLt, indexIn64Lt, RoundMode::CAST_NONE, thisLen);

            indexInQue.FreeTensor(indexIn64Lt);
#endif

            Muls(indexInLt, indexInLt, static_cast<int32_t>(sizeof(xType)), thisLen);
            LocalTensor<uint32_t> indexUInt32Lt = indexInLt.ReinterpretCast<uint32_t>();
            LocalTensor<xType> outLt = outQue.AllocTensor<xType>();
            Gather<xType>(outLt, xLt, indexUInt32Lt, static_cast<uint32_t>(0), paddingLen);

            outQue.EnQue(outLt);
#ifdef SUPPORT_V200
            indexInQue.FreeTensor(indexInLt);
#endif
            LocalTensor<xType> outNewLt = outQue.DeQue<xType>();
            pipe_barrier(PIPE_ALL);

            CpLocal2Gm(yGt[thisOffset], outNewLt, thisLen);
            outQue.FreeTensor(outNewLt);
            remain = remain - thisLen;
        }
    }

    __aicore__ inline void Compute()
    {
#ifndef SUPPORT_V200
        Duplicate<int32_t>(indexCastUint32Buf.Get<int32_t>(), 0, oneBlockElem);
#endif
        int64_t coreLen = indexDim0 / GetBlockNum();
        int64_t coreSplitId = indexDim0 % GetBlockNum();

        int64_t lenOfThisCore = 0;
        int64_t offsetOfThisCore = 0;

        if (GetBlockIdx() >= coreSplitId) {
            lenOfThisCore = coreLen;
            offsetOfThisCore = coreSplitId * (coreLen + 1) + (GetBlockIdx() - coreSplitId) * coreLen;
        } else {
            lenOfThisCore = coreLen + 1;
            offsetOfThisCore = GetBlockIdx() * (coreLen + 1);
        }

        LocalTensor<xType> xInLt = xInQue.AllocTensor<xType>();
        CpPadGm2Local(xInLt, xGt, xDim0);
        xInQue.EnQue(xInLt);
        xInLt = xInQue.DeQue<xType>();
        pipe_barrier(PIPE_ALL);

        GatherX(xInLt, offsetOfThisCore, lenOfThisCore);
        xInQue.FreeTensor(xInLt);
    }

private:
    // GM_ADDR
    GM_ADDR x;
    GM_ADDR index;
    GM_ADDR y;

    // Shape
    int64_t xDim0;
    int64_t indexDim0;
    int64_t ubCanUsed;

    // Tiling
    int64_t oneBlockElem;

    // Gt
    GlobalTensor<xType> xGt;
#ifdef SUPPORT_V200
    GlobalTensor<int32_t> indexGt;
#else
    GlobalTensor<int64_t> indexGt;
#endif
    GlobalTensor<xType> yGt;

    // TPipe
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xInQue;
    TQue<QuePosition::VECIN, 1> indexInQue;
    TQue<QuePosition::VECOUT, 1> outQue;
#ifndef SUPPORT_V200
    TBuf<TPosition::VECCALC> indexCastUint32Buf;
#endif
};
}  // namespace GatherForRank1
#endif
