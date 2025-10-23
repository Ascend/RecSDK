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
#include "hstu_dense_forward_fuxi_tiling.h"
#include "ops_log.h"
#include "matmul_check.h"
using namespace MatmulTilingCheck;

namespace HstuDenseForwardFuxi {

bool QKVShapeCheck(gert::TilingContext* context, int qkvDim);

class ShapeRange {
public:
    int64_t lbound {0}; // shape下限
    int64_t ubound {0}; // shape上限
    int64_t mutiple {0}; // 倍数
    const char* name {nullptr};
    ShapeRange(int64_t lbound, int64_t ubound, int64_t mutiple, const char* name);
    bool Check(int64_t val) const;
};

class TilingPolicy {
public:
    virtual ge::graphStatus InferShape(gert::InferShapeContext* context);

    virtual ge::graphStatus InferDtype(gert::InferDataTypeContext* context);

    virtual ge::graphStatus TilingProcess(gert::TilingContext* context);

    virtual bool GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim);

private:
    virtual bool CheckIsSupport(gert::TilingContext* context);

    virtual bool TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling);

    virtual bool TilingAttribute(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling);

    virtual bool TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling);

    virtual bool TilingHeighLevelApi(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling);

    virtual bool TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling);

    virtual bool TilingSaveToBuffer(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling);
};

}

#endif