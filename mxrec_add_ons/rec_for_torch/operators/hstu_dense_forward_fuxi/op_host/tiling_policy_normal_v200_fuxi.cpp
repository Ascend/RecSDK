#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling_policy_define.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_normal_v200_fuxi.h"

namespace HstuDenseForwardFuxi {
REGISTER_POLICY(LAYOUT_TYPE::NORMALV200, std::make_shared<TilingPolicyNormalv200Fuxi>());

int GetMaxTmpUbSize(int qkTmpSize, int svTmpSize, int tvTmpSize, int pvTmpSize)
{
    int tmpUbSize = qkTmpSize > svTmpSize ? qkTmpSize : svTmpSize;
    tmpUbSize = tmpUbSize > tvTmpSize ? tmpUbSize : tvTmpSize;
    tmpUbSize = tmpUbSize > pvTmpSize ? tmpUbSize : pvTmpSize;
    return tmpUbSize;
}

bool TilingPolicyNormalv200Fuxi::GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim)
{
    static const ShapeRange SEQ_RANGE(64, 20480, BLOCK_HEIGHT, "seq size");
    static const ShapeRange BATCH_RANGE(1, MAX_BATCH_SIZE, 1, "batch size");
    static const ShapeRange DIM_RANGE(16, 128, 16, "dim size");
    static const ShapeRange HEAD_RANGE(2, 8, 2, "head num");

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

ge::graphStatus TilingPolicyNormalv200Fuxi::InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(INDEX_T::INDEX_0);
    OPS_LOG_E_IF_NULL("qShape", qShape, return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    outputShape->SetDimNum(OUTPUT_DIM_NUM);

    outputShape->SetDim(INDEX_T::INDEX_0, qShape->GetDim(INDEX_T::INDEX_0));
    outputShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(INDEX_T::INDEX_1));

    // 获取算子可选参数：timestampBias和positionBias
    auto timestampBias = context->GetOptionalInputTensor(INDEX_T::INDEX_3);
    auto positionBias = context->GetOptionalInputTensor(INDEX_T::INDEX_4);
    if ((timestampBias == nullptr) || (positionBias == nullptr)) {
        outputShape->SetDim(INDEX_T::INDEX_2, qShape->GetDim(INDEX_T::INDEX_2) * qShape->GetDim(INDEX_T::INDEX_3))
    } else {
        // 3:attnOut + tsOut + posOut
        outputShape->SetDim(INDEX_T::INDEX_2, 3 * qShape->GetDim(INDEX_T::INDEX_2) * qShape->GetDim(INDEX_T::INDEX_3))
    }

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicyNormalv200Fuxi::TilingShape(gert::TilingContext* context,
    optiling::HstuDenseForwardFuxiTilingData &tiling)
{
    auto qShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();

    int64_t batchSize = qShape.GetDim(INDEX_T::INDEX_0);
    tiling.set_batchSize(batchSize);
    int64_t seqLen = qShape.GetDim(INDEX_T::INDEX_1);
    tiling.set_seqLen(seqLen);
    int64_t headNum = qShape.GetDim(INDEX_T::INDEX_2);
    tiling.set_headNum(headNum);
    int64_t dim = qShape.GetDim(INDEX_T::INDEX_3);
    tiling.set_dim(dim);

    OPS_LOG_E_IF(!GeneralShapeCheck(batchSize, seqLen, headNum, dim), context, return false, "Shape check failed");
    return true;
}

bool TilingPolicyNormalv200Fuxi::TilingMatmul(gert::TilingContext* context,
    optiling::HstuDenseForwardFuxiTilingData &tiling, matmul_tiling::DataType dataType)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int64_t dim = tiling.get_dim();
    tiling.set_blockHeight(BLOCK_HEIGHT);

    // apply qk
    matmul_tiling::MatmulApiTiling qkMatmul(ascendPlatform);
    qkMatmul.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
    qkMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    qkMatmul.SetOrgShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetBias(false);
    qkMatmul.SetBufferSpace(-1, -1, -1);

    // sv
    matmul_tiling::MatmulApiTiling svMatmul(ascendPlatform);
    svMatmul.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    svMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    svMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    svMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    svMatmul.SetBias(false);
    svMatmul.SetBufferSpace(-1, -1, -1);

    // tv
    matmul_tiling::MatmulApiTiling tvMatmul(ascendPlatform);
    tvMatmul.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    tvMatmul.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    tvMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tvMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    tvMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    tvMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    tvMatmul.SetBias(false);
    tvMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling pvMatmul(ascendPlatform);
    pvMatmul.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    pvMatmul.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND, dataType);
    pvMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    pvMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    pvMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    pvMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    pvMatmul.SetBias(false);
    pvMatmul.SetBufferSpace(-1, -1, -1);

    if ((qkMatmul.GetTiling(tiling.qkMatmul) == -1) || (svMatmul.GetTiling(tiling.svMatmul) == -1) ||
        (tvMatmul.GetTiling(tiling.tvMatmul) == -1) || (pvMatmul.GetTiling(tiling.pvMatmul) == -1)) {
        OPS_LOG_E("GetTiling failed.\n");
        return false;
    }

    return true;
}

bool TilingPolicyNormalv200Fuxi::TilingHeighLevelApi(gert::TilingContext* context,
    optiling::HstuDenseForwardFuxiTilingData &tiling)
{
    int64_t dim = tiling.get_dim();

    matmul_tiling::DataType dataType;
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * dim * COMPUTE_PIPE_NUM;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t workspaceSize = oneCoreMidElem * sizeof(float);
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize;

    OPS_LOG_E_IF(!TilingMatmul(context, tiling, dataType), context, return false, "TilingMatmul failed");

    int qkTransLength = tiling.qkMatmul.get_transLength();
    int svTransLength = tiling.svMatmul.get_transLength();
    int tvTransLength = tiling.tvMatmul.get_transLength();
    int pvTransLength = tiling.pvMatmul.get_transLength();

    int transLength = GetMaxTmpUbSize(qkTransLength, svTransLength, tvTransLength, pvTransLength);
    tiling.set_tmpUbSize(transLength);

    tiling.set_qkBaseM(tiling.qkMatmul.get_baseM());
    tiling.set_qkBaseN(tiling.qkMatmul.get_baseN());

    tiling.set_svBaseM(tiling.svMatmul.get_baseM());
    tiling.set_svBaseN(tiling.svMatmul.get_baseN());

    tiling.set_tvBaseM(tiling.tvMatmul.get_baseM());
    tiling.set_tvBaseN(tiling.tvMatmul.get_baseN());

    tiling.set_pvBaseM(tiling.pvMatmul.get_baseM());
    tiling.set_pvBaseN(tiling.pvMatmul.get_baseN());

    return true;
}

bool TilingPolicyNormalv200Fuxi::TilingKeySet(gert::TilingContext* context,
    optiling::HstuDenseForwardFuxiTilingData &tiling)
{
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(FLOAT16_TILING_KEY);
    } else {
        OPS_LOG_E("invalid datatype, only support fp16.\n");
        return false;
    }

    return true;
}
}