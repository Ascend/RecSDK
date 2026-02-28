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


#ifndef TILING_POLICY_H
#define TILING_POLICY_H

#include "register/op_def_registry.h"

namespace HstuForward {

bool QKVShapeCheck(gert::TilingContext* context, int qkvDim);


struct TilingKeyParam {
    bool enableBias;
    bool deterministic;
    uint32_t maskType;
    int64_t dimQ;
    int64_t dimV;
    int64_t maxSeqLenQ;
    int64_t maxSeqLenK;
};

class ShapeRange {
public:
    int64_t lbound {0}; // shape下限
    int64_t ubound {0}; // shape上限
    int64_t multiple {0}; // 倍数
    const char* name {nullptr};
    ShapeRange(int64_t lbound, int64_t ubound, int64_t multiple, const char* name);
    bool Check(int64_t val) const;
};

class TilingPolicy {
public:
    virtual ge::graphStatus InferShape(gert::InferShapeContext* context);

    virtual ge::graphStatus InferDtype(gert::InferDataTypeContext* context);

    virtual ge::graphStatus TilingProcess(gert::TilingContext* context);

    virtual bool GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim,
        bool dimAlign = false);

    virtual bool TilingWorkSpace(gert::TilingContext* context);

    virtual bool TilingKeySetImpl(gert::TilingContext* context, const TilingKeyParam& tilingKeyParam,
        uint32_t typeTilingKey);

    virtual bool TilingCore(gert::TilingContext* context);
};

}

#endif