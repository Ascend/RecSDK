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

#include "tiling_policy.h"
#include <cstdint>
#include "register/op_def_registry.h"
#include "matmul_check.h"
#include "../op_kernel/hstu_kernel_tiling_key.h"
#include "tiling_policy_define.h"

using namespace MatmulTilingCheck;

namespace HstuForward {

ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t multiple, const char* name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->multiple = multiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_CHECK((val < lbound || val > ubound || val % multiple != 0),
              OPS_LOG_E("", "%s must meet range[%lld %lld] and multiple of [%lld]. but get value %lld\n", name, lbound,
                        ubound, multiple, val),
              return false);
    return true;
}

ge::graphStatus TilingPolicy::InferShape(gert::InferShapeContext* context)
{
    const gert::Shape *queryShape = context->GetInputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(queryShape, return ge::GRAPH_FAILED);

    gert::Shape *attenOutputShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(attenOutputShape, return ge::GRAPH_FAILED);
    *attenOutputShape = *queryShape;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPolicy::TilingProcess(gert::TilingContext* context)
{
    return ge::GRAPH_SUCCESS;
}

bool TilingPolicy::GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim, bool dimAlign)
{
    auto dimStep = dimAlign ? 16 : 1; // 16 means align to C0_size 1 means not align
    const ShapeRange SEQ_RANGE(1, 20480, 1, "seq size");
    const ShapeRange BATCH_RANGE(1, MAX_BATCH_SIZE, 1, "batch size");
    const ShapeRange DIM_RANGE(dimStep, 512, dimStep, "dim size"); // 非对齐情况下dim范围[1,512]
    const ShapeRange HEAD_RANGE(1, 16, 1, "head num");

    if (!SEQ_RANGE.Check(seqLen)) {
        return false;
    }

    if (!BATCH_RANGE.Check(batchSize)) {
        return false;
    }

    if (!HEAD_RANGE.Check(headNum)) {
        return false;
    }

    if (!DIM_RANGE.Check(dim)) {
        return false;
    }

    return true;
}

bool TilingPolicy::TilingWorkSpace(gert::TilingContext* context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return false);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * MAX_DIM * TRANS_PIPE_NUM;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float);
    auto socVersion = ascendPlatform.GetSocVersion();
    // A5平台需要额外的FP8 workspace
    if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
        workspaceSize += oneCoreMidElem * 1;
    }
    int64_t syncSize = coreNum * VCORE_NUM_IN_ONE_AIC * DATA_ALIGN_BYTES;
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize + syncSize;
    return true;
}

bool TilingPolicy::TilingCore(gert::TilingContext* context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum > MAX_AIV_NUM, OPS_LOG_E("", "vecCoreNum %d should be < %d\n", coreNum, MAX_AIV_NUM),
              return false);
    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    return true;
}

#ifdef SUPPORT_950
void Find4BytesShape(const TilingKeyParam& tilingKeyParam, uint32_t &tilingM, uint32_t &tilingN)
{
    constexpr uint32_t tileMList[] = {32, 64, 64, 128, 128, 256, 128};
    constexpr uint32_t tileNList[] = {32, 64, 1024, 128, 512, 256, 256};
    int idx = NO_TILING_IDX;
    int area = MAX_DIM * MAX_DIM;
    auto maxSeqLenQ = tilingKeyParam.maxSeqLenQ;
    auto maxSeqLenK = tilingKeyParam.maxSeqLenK;
    if (maxSeqLenQ > maxSeqLenK) {
        idx = (maxSeqLenK >= TILING_1024) ? 5 :
        (maxSeqLenK >= TILING_64)   ? 3 :
        (maxSeqLenK >= TILING_32)   ? 1 : 0;
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    
    for (int i = 0; i < TILING_SIZE; i++) {
        if (maxSeqLenQ <= tileMList[i] && maxSeqLenK <= tileNList[i]) {
            if (tileMList[i] * tileNList[i] < area && maxSeqLenK > tileNList[i] / 2) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
    }
    if (idx != NO_TILING_IDX) {
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }

    double ratio = (double)maxSeqLenK / maxSeqLenQ;
    idx = 0;
    double delta = fabs((double)tileNList[idx] / tileMList[idx] - ratio);
    area = tileMList[idx] * tileNList[idx];
    for (int i = 1; i < TILING_SIZE; i++) {
        double tmp = fabs((double)tileNList[i] / tileMList[i] - ratio);
        if (tmp < delta || (tmp == delta && area < tileMList[i] * tileNList[i])) {
            idx = i;
            delta = tmp;
            area = tileMList[idx] * tileNList[idx];
        }
    }
    tilingM = tileMList[idx];
    tilingN = tileNList[idx];
    return;
}


void Find2BytesShape(const TilingKeyParam& tilingKeyParam, uint32_t &tilingM, uint32_t &tilingN)
{
    constexpr uint32_t tileMList[] = {32, 64, 64, 128, 128, 256, 128};
    constexpr uint32_t tileNList[] = {32, 64, 1024, 128, 512, 256, 256};
    int idx = NO_TILING_IDX;
    int area = MAX_DIM * MAX_DIM;
    auto maxSeqLenQ = tilingKeyParam.maxSeqLenQ;
    auto maxSeqLenK = tilingKeyParam.maxSeqLenK;
    if (maxSeqLenQ > maxSeqLenK) {
        if (maxSeqLenK > TILING_128) {
            idx = 5;
        } else if (maxSeqLenK > TILING_64) {
            idx = 3;
        } else if (maxSeqLenK > TILING_32) {
            idx = 1;
        } else {
            idx = 0;
        }
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    for (int i = 0; i < TILING_SIZE; i++) {
        if (maxSeqLenQ <= tileMList[i] && maxSeqLenK <= tileNList[i]) {
            if (tileMList[i] * tileNList[i] < area && maxSeqLenK > tileNList[i] / 2) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
    }
    if (idx != NO_TILING_IDX) {
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }

    for (int i = 0; i < TILING_SIZE; i++) {
        if (maxSeqLenQ <= tileMList[i] * 4 && maxSeqLenK <= tileNList[i]) {
            if (tileMList[i] * tileNList[i] < area) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
    }
    if (idx != NO_TILING_IDX) {
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    if (maxSeqLenQ >= TILING_512 && maxSeqLenK >= TILING_1024) {
        idx = 5;
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    if (maxSeqLenQ >= TILING_256 && maxSeqLenK >= TILING_512) {
        idx = 4;
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    if (maxSeqLenK < maxSeqLenQ * 2) {
        if (maxSeqLenK >= TILING_256) {
            idx = 5;
            tilingM = tileMList[idx];
            tilingN = tileNList[idx];
            return;
        }
        for (int i = 0; i < TILING_SIZE; i++) {
            if (tileMList[i] != tileNList[i] || tileNList[i] < maxSeqLenK) {
                continue;
            }
            if (tileMList[i] * tileNList[i] < area) {
                idx = i;
                area = tileMList[i] * tileNList[i];
            }
        }
        tilingM = tileMList[idx];
        tilingN = tileNList[idx];
        return;
    }
    idx = 6;
    tilingM = tileMList[idx];
    tilingN = tileNList[idx];
    return;
}

void FindMatchShape(const TilingKeyParam& tilingKeyParam,
    gert::TilingContext* context, uint32_t &tilingM, uint32_t &tilingN, uint32_t &tilingDim)
{
    constexpr uint32_t tileKList[] = {64, 128, 256, 512};

    uint32_t vdim = tilingKeyParam.dimV;
    uint32_t qdim = tilingKeyParam.dimQ;
    tilingDim = *std::lower_bound(std::begin(tileKList), std::end(tileKList), std::max(qdim, vdim));
    ge::DataType qTypeGe = context->GetInputTensor(JAGGED_INPUT_INDEX_T::Q_INDEX)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        Find4BytesShape(tilingKeyParam, tilingM, tilingN);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16 || qTypeGe == ge::DataType::DT_BF16) {
        Find2BytesShape(tilingKeyParam, tilingM, tilingN);
    }
    return;
}

bool TilingPolicy::TilingKeySetImpl(gert::TilingContext* context, const TilingKeyParam& tilingKeyParam,
    uint32_t typeTilingKey)
{
    bool enableBias = tilingKeyParam.enableBias;
    bool enableDeteministic = tilingKeyParam.deterministic;
    uint32_t maskType = tilingKeyParam.maskType;
    uint32_t maskedType = maskType & 0x3;
    uint32_t tilingM = 0;
    uint32_t tilingN = 0;
    uint32_t tilingDim = 0;
    
    ge::DataType qTypeGe = context->GetInputTensor(JAGGED_INPUT_INDEX_T::Q_INDEX)->GetDataType();
    if ((qTypeGe == ge::DataType::DT_FLOAT8_E4M3FN) || (typeTilingKey != (JAGGED_TILING_KEY & 0x3))) {
        tilingDim = MAX_DIM;
        tilingM = MAX_TILING_DIM;
        tilingN = MAX_TILING_DIM;
    } else {
        FindMatchShape(tilingKeyParam, context, tilingM, tilingN, tilingDim);
    }
    
    const uint64_t tilingKey = GET_TPL_TILING_KEY(maskedType, enableBias,
                                                  typeTilingKey, enableDeteministic, tilingM,
                                                  tilingN, tilingDim);
    context->SetTilingKey(tilingKey);
    return true;
}
#else
bool TilingPolicy::TilingKeySetImpl(gert::TilingContext* context, const TilingKeyParam& tilingKeyParam,
    uint32_t typeTilingKey)
{
    bool enableBias = tilingKeyParam.enableBias;
    bool enableDeteministic = tilingKeyParam.deterministic;
    uint32_t maskType = tilingKeyParam.maskType;
    uint32_t maskedType = maskType & 0x3;
    // 组合tiling key：
    
    const uint64_t tilingKey = GET_TPL_TILING_KEY(maskedType, enableBias,
                                                  typeTilingKey, enableDeteministic, MAX_TILING_DIM,
                                                  MAX_TILING_DIM, MAX_DIM);
    context->SetTilingKey(tilingKey);

    return true;
}
#endif

}
