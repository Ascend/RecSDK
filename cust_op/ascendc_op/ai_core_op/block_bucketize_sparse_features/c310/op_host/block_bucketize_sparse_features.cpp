/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#include <cstdint>
#include <limits>

#include "block_bucketize_sparse_features_tiling.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int32_t EXPECTED_RANK = 1;
constexpr size_t LOCAL_MEMORY_SIZE = 216 * 1024;

template <typename UnsignedT>
inline void HostPrecomputeFastDivmod(UnsignedT divisor, UnsignedT& outMagic, uint32_t& outShift)
{
    if (divisor <= 1) {
        outMagic = 0;
        outShift = 0;
        return;
    }
    unsigned __int128 one = 1;
    constexpr uint32_t BIT_WIDTH = static_cast<uint32_t>(sizeof(UnsignedT) * 8); /* 单字节 8 BIT */
    uint32_t s = 0;
    for (; s < BIT_WIDTH; ++s) {
        if ((one << s) >= static_cast<unsigned __int128>(divisor)) {
            break;
        }
    }
    outShift = s;
    outMagic = static_cast<UnsignedT>(
        ((one << BIT_WIDTH) * ((one << s) - static_cast<unsigned __int128>(divisor))) /
        static_cast<unsigned __int128>(divisor) + 1);
}

enum ComputeNewLengthsInputIndex : int32_t {
    CNL_INPUT_INDICES_INDEX = 0,
    CNL_INPUT_BLOCK_SIZES_INDEX = 1,
    CNL_INPUT_OFFSETS_INDEX = 2,
    CNL_INPUT_TOTAL_NUM_BLOCKS_INDEX = 3,
    CNL_INPUT_BATCH_SIZE_OFFSETS_INDEX = 4,
    CNL_INPUT_BLOCK_BUCKETIZE_POS_INDEX = 5,
};

enum ComputeNewLengthsOutputIndex : int32_t {
    CNL_OUTPUT_NEW_LENGTHS_INDEX = 0,
};

enum ComputeNewLengthsAttrIndex : int32_t {
    CNL_ATTR_MY_SIZE_INDEX = 0,
    CNL_ATTR_BUCKETIZE_POS_INDEX = 1,
    CNL_ATTR_LENGTHS_SIZE_INDEX = 2,
    CNL_ATTR_BATCH_SIZE_INDEX = 3,
    CNL_ATTR_MAX_B_INDEX = 4,
};

enum ScatterInputIndex : int32_t {
    SNI_INPUT_INDICES_INDEX = 0,
    SNI_INPUT_BLOCK_SIZES_INDEX = 1,
    SNI_INPUT_OFFSETS_INDEX = 2,
    SNI_INPUT_NEW_OFFSETS_INDEX = 3,
    SNI_INPUT_WEIGHTS_INDEX = 4,
    SNI_INPUT_TOTAL_NUM_BLOCKS_INDEX = 5,
    SNI_INPUT_BATCH_SIZE_OFFSETS_INDEX = 6,
    SNI_INPUT_BLOCK_BUCKETIZE_POS_INDEX = 7,
};

enum ScatterOutputIndex : int32_t {
    SNI_OUTPUT_NEW_INDICES_INDEX = 0,
    SNI_OUTPUT_NEW_WEIGHTS_INDEX = 1,
    SNI_OUTPUT_NEW_POS_INDEX = 2,
    SNI_OUTPUT_UNBUCKETIZE_PERMUTE_INDEX = 3,
};

enum ScatterAttrIndex : int32_t {
    SNI_ATTR_MY_SIZE_INDEX = 0,
    SNI_ATTR_BUCKETIZE_POS_INDEX = 1,
    SNI_ATTR_SEQUENCE_INDEX = 2,
    SNI_ATTR_KEEP_ORIG_IDX_INDEX = 3,
    SNI_ATTR_LENGTHS_SIZE_INDEX = 4,
    SNI_ATTR_BATCH_SIZE_INDEX = 5,
    SNI_ATTR_MAX_B_INDEX = 6,
};

constexpr int64_t TILING_KEY_FULL = 0;
constexpr int64_t TILING_KEY_SIMPLIFIED = 1;

} // namespace

namespace optiling {


void FillCommonTilingFields(
    BlockBucketizeSparseFeaturesTilingData& tiling,
    int64_t lengthsSize,
    int64_t indicesSize,
    int64_t numFeatures,
    int64_t batchSize,
    int64_t mySize,
    int64_t maxB,
    bool enableBucketizePos,
    bool enableTotalNumBlocks,
    bool enableBatchSizePerFeature,
    bool hasBucketizePosList,
    ge::DataType indicesDataType)
{
    tiling.set_lengthsSize(lengthsSize);
    tiling.set_indicesSize(indicesSize);
    tiling.set_numFeatures(numFeatures);
    tiling.set_batchSize(batchSize);
    tiling.set_mySize(mySize);
    tiling.set_newLengthsSize(lengthsSize * mySize);
    tiling.set_maxBatchSize(maxB);
    tiling.set_enableBucketizePos(enableBucketizePos);
    tiling.set_enableTotalNumBlocks(enableTotalNumBlocks);
    tiling.set_enableBatchSizePerFeature(enableBatchSizePerFeature);
    tiling.set_enableBlockBucketizePos(hasBucketizePosList);

    uint64_t mySizeMagic = 0;
    uint32_t mySizeShift = 0;
    if (indicesDataType == ge::DT_INT32) {
        uint32_t mySizeMagic32 = 0;
        HostPrecomputeFastDivmod<uint32_t>(static_cast<uint32_t>(mySize), mySizeMagic32, mySizeShift);
        mySizeMagic = mySizeMagic32;
    } else {
        HostPrecomputeFastDivmod<uint64_t>(static_cast<uint64_t>(mySize), mySizeMagic, mySizeShift);
    }
    tiling.set_mySizeDivMagic(mySizeMagic);
    tiling.set_mySizeDivShift(mySizeShift);

    uint64_t batchSizeMagic = 0;
    uint32_t batchSizeShift = 0;
    if (batchSize > 0) {
        HostPrecomputeFastDivmod<uint64_t>(static_cast<uint64_t>(batchSize), batchSizeMagic, batchSizeShift);
    }
    tiling.set_batchSizeDivMagic(batchSizeMagic);
    tiling.set_batchSizeDivShift(batchSizeShift);
}

static ge::graphStatus ComputeNewLengthsTilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(CNL_INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesTensor", context->GetInputTensor(CNL_INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("blockSizesShape", context->GetInputShape(CNL_INPUT_BLOCK_SIZES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(CNL_INPUT_OFFSETS_INDEX), return ge::GRAPH_FAILED);

    const auto* indicesShape = context->GetInputShape(CNL_INPUT_INDICES_INDEX);
    const auto* blockSizesShape = context->GetInputShape(CNL_INPUT_BLOCK_SIZES_INDEX);
    const auto* totalNumBlocksShape = context->GetInputShape(CNL_INPUT_TOTAL_NUM_BLOCKS_INDEX);
    const auto* posTensorShape = context->GetInputShape(CNL_INPUT_BLOCK_BUCKETIZE_POS_INDEX);
    const auto* batchSizeOffsetsShape = context->GetInputShape(CNL_INPUT_BATCH_SIZE_OFFSETS_INDEX);

    const auto indicesStorageShape = indicesShape->GetStorageShape();
    const auto blockSizesStorageShape = blockSizesShape->GetStorageShape();
    const int64_t indicesSize = indicesStorageShape.GetShapeSize();
    const int64_t numFeatures = blockSizesStorageShape.GetShapeSize();
    const auto indicesDataType = context->GetInputTensor(CNL_INPUT_INDICES_INDEX)->GetDataType();

    auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* mySizePtr = attrs->GetAttrPointer<int64_t>(CNL_ATTR_MY_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("my_size attr", mySizePtr, return ge::GRAPH_FAILED);
    const int64_t mySize = *mySizePtr;

    const bool* bucketizePosPtr = attrs->GetAttrPointer<bool>(CNL_ATTR_BUCKETIZE_POS_INDEX);
    OPS_LOG_E_IF_NULL("bucketize_pos attr", bucketizePosPtr, return ge::GRAPH_FAILED);

    const int64_t* lengthsSizePtr = attrs->GetAttrPointer<int64_t>(CNL_ATTR_LENGTHS_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("lengths_size attr", lengthsSizePtr, return ge::GRAPH_FAILED);
    const int64_t lengthsSize = *lengthsSizePtr;

    const int64_t* batchSizePtr = attrs->GetAttrPointer<int64_t>(CNL_ATTR_BATCH_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("batch_size attr", batchSizePtr, return ge::GRAPH_FAILED);
    const int64_t batchSize = *batchSizePtr;

    const int64_t* maxBPtr = attrs->GetAttrPointer<int64_t>(CNL_ATTR_MAX_B_INDEX);
    OPS_LOG_E_IF_NULL("max_B attr", maxBPtr, return ge::GRAPH_FAILED);
    const int64_t maxB = *maxBPtr;

    const bool enableTotalNumBlocks = (totalNumBlocksShape != nullptr);
    const bool enableBatchSizePerFeature = (batchSizeOffsetsShape != nullptr);

    bool hasBucketizePosList = false;
    if (posTensorShape != nullptr) {
        hasBucketizePosList = (posTensorShape->GetStorageShape().GetShapeSize() > 0);
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspace", workspace, return ge::GRAPH_FAILED);
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();

    constexpr uint64_t ALIGN64 = 64;
    uint64_t nextOffset = 0;

    uint64_t posPtrsOffsetVal = 0;
    uint64_t posLensOffsetVal = 0;
    if (hasBucketizePosList) {
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        posPtrsOffsetVal = nextOffset;
        nextOffset += static_cast<uint64_t>(numFeatures) * sizeof(uint64_t);
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        posLensOffsetVal = nextOffset;
        nextOffset += static_cast<uint64_t>(numFeatures) * sizeof(int64_t);
    }

    workspace[0] = systemWorkspaceSize + nextOffset;

    constexpr int64_t KERNEL_WARPS_PER_BLOCK = 16;
    const int64_t blocksForRows = (lengthsSize + KERNEL_WARPS_PER_BLOCK - 1) / KERNEL_WARPS_PER_BLOCK;
    int64_t totalBlocksMax = (blocksForRows > 0) ? blocksForRows : 1;

    const size_t maxCores = ascendPlatform.GetCoreNumAiv();
    size_t coreNum = (totalBlocksMax < static_cast<int64_t>(maxCores)) ?
        static_cast<size_t>(totalBlocksMax) : maxCores;
    coreNum = (coreNum == 0) ? 1 : coreNum;

    const bool useSimplified = !enableTotalNumBlocks && !enableBatchSizePerFeature && !hasBucketizePosList;

    BlockBucketizeSparseFeaturesTilingData tiling;
    FillCommonTilingFields(tiling, lengthsSize, indicesSize, numFeatures,
        batchSize, mySize, maxB, *bucketizePosPtr, enableTotalNumBlocks,
        enableBatchSizePerFeature, hasBucketizePosList, indicesDataType);
    tiling.set_enableSequence(false);
    tiling.set_enableWeights(false);
    tiling.set_enableKeepOrigIdx(false);
    tiling.set_posPtrsOffset(posPtrsOffsetVal);
    tiling.set_posLensOffset(posLensOffsetVal);

    context->SetTilingKey(useSimplified ? TILING_KEY_SIMPLIFIED : TILING_KEY_FULL);
    context->SetBlockDim(coreNum);
    context->SetLocalMemorySize(LOCAL_MEMORY_SIZE);
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterNewIndicesTilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(SNI_INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesTensor", context->GetInputTensor(SNI_INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("blockSizesShape", context->GetInputShape(SNI_INPUT_BLOCK_SIZES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("offsetsShape", context->GetInputShape(SNI_INPUT_OFFSETS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("newOffsetsShape", context->GetInputShape(SNI_INPUT_NEW_OFFSETS_INDEX), return ge::GRAPH_FAILED);

    const auto* indicesShape = context->GetInputShape(SNI_INPUT_INDICES_INDEX);
    const auto* blockSizesShape = context->GetInputShape(SNI_INPUT_BLOCK_SIZES_INDEX);
    const auto* totalNumBlocksShape = context->GetInputShape(SNI_INPUT_TOTAL_NUM_BLOCKS_INDEX);
    const auto* posTensorShape = context->GetInputShape(SNI_INPUT_BLOCK_BUCKETIZE_POS_INDEX);
    const auto* batchSizeOffsetsShape = context->GetInputShape(SNI_INPUT_BATCH_SIZE_OFFSETS_INDEX);

    const auto indicesStorageShape = indicesShape->GetStorageShape();
    const auto blockSizesStorageShape = blockSizesShape->GetStorageShape();
    const int64_t indicesSize = indicesStorageShape.GetShapeSize();
    const int64_t numFeatures = blockSizesStorageShape.GetShapeSize();
    const auto indicesDataType = context->GetInputTensor(SNI_INPUT_INDICES_INDEX)->GetDataType();

    auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* mySizePtr = attrs->GetAttrPointer<int64_t>(SNI_ATTR_MY_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("my_size attr", mySizePtr, return ge::GRAPH_FAILED);
    const int64_t mySize = *mySizePtr;

    const bool* bucketizePosPtr = attrs->GetAttrPointer<bool>(SNI_ATTR_BUCKETIZE_POS_INDEX);
    OPS_LOG_E_IF_NULL("bucketize_pos attr", bucketizePosPtr, return ge::GRAPH_FAILED);

    const bool* sequencePtr = attrs->GetAttrPointer<bool>(SNI_ATTR_SEQUENCE_INDEX);
    OPS_LOG_E_IF_NULL("sequence attr", sequencePtr, return ge::GRAPH_FAILED);

    const bool* keepOrigIdxPtr = attrs->GetAttrPointer<bool>(SNI_ATTR_KEEP_ORIG_IDX_INDEX);
    OPS_LOG_E_IF_NULL("keep_orig_idx attr", keepOrigIdxPtr, return ge::GRAPH_FAILED);

    const int64_t* lengthsSizePtr = attrs->GetAttrPointer<int64_t>(SNI_ATTR_LENGTHS_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("lengths_size attr", lengthsSizePtr, return ge::GRAPH_FAILED);
    const int64_t lengthsSize = *lengthsSizePtr;

    const int64_t* batchSizePtr = attrs->GetAttrPointer<int64_t>(SNI_ATTR_BATCH_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("batch_size attr", batchSizePtr, return ge::GRAPH_FAILED);
    const int64_t batchSize = *batchSizePtr;

    const int64_t* maxBPtr = attrs->GetAttrPointer<int64_t>(SNI_ATTR_MAX_B_INDEX);
    OPS_LOG_E_IF_NULL("max_B attr", maxBPtr, return ge::GRAPH_FAILED);
    const int64_t maxB = *maxBPtr;

    const bool enableTotalNumBlocks = (totalNumBlocksShape != nullptr);
    const bool enableBatchSizePerFeature = (batchSizeOffsetsShape != nullptr);

    bool hasBucketizePosList = false;
    if (posTensorShape != nullptr) {
        hasBucketizePosList = (posTensorShape->GetStorageShape().GetShapeSize() > 0);
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspace", workspace, return ge::GRAPH_FAILED);
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();

    constexpr uint64_t ALIGN64 = 64;
    uint64_t nextOffset = 0;

    uint64_t posPtrsOffsetVal = 0;
    uint64_t posLensOffsetVal = 0;
    if (hasBucketizePosList) {
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        posPtrsOffsetVal = nextOffset;
        nextOffset += static_cast<uint64_t>(numFeatures) * sizeof(uint64_t);
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        posLensOffsetVal = nextOffset;
        nextOffset += static_cast<uint64_t>(numFeatures) * sizeof(int64_t);
    }

    workspace[0] = systemWorkspaceSize + nextOffset;

    constexpr int64_t KERNEL_WARPS_PER_BLOCK = 16;
    const int64_t blocksForRows = (lengthsSize + KERNEL_WARPS_PER_BLOCK - 1) / KERNEL_WARPS_PER_BLOCK;
    int64_t totalBlocksMax = (blocksForRows > 0) ? blocksForRows : 1;

    const size_t maxCores = ascendPlatform.GetCoreNumAiv();
    size_t coreNum = (totalBlocksMax < static_cast<int64_t>(maxCores)) ?
        static_cast<size_t>(totalBlocksMax) : maxCores;
    coreNum = (coreNum == 0) ? 1 : coreNum;

    const auto* weightsTensor = context->GetInputTensor(SNI_INPUT_WEIGHTS_INDEX);
    const bool useSimplified = !enableTotalNumBlocks && !enableBatchSizePerFeature &&
                               !hasBucketizePosList && !(*keepOrigIdxPtr);

    BlockBucketizeSparseFeaturesTilingData tiling;
    FillCommonTilingFields(tiling, lengthsSize, indicesSize, numFeatures,
        batchSize, mySize, maxB, *bucketizePosPtr, enableTotalNumBlocks,
        enableBatchSizePerFeature, hasBucketizePosList, indicesDataType);
    tiling.set_enableSequence(*sequencePtr);
    tiling.set_enableWeights(weightsTensor != nullptr);
    tiling.set_enableKeepOrigIdx(*keepOrigIdxPtr);
    tiling.set_posPtrsOffset(posPtrsOffsetVal);
    tiling.set_posLensOffset(posLensOffsetVal);

    context->SetTilingKey(useSimplified ? TILING_KEY_SIMPLIFIED : TILING_KEY_FULL);
    context->SetBlockDim(coreNum);
    context->SetLocalMemorySize(LOCAL_MEMORY_SIZE);
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus ComputeNewLengthsInferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* mySizePtr = attrs->GetAttrPointer<int64_t>(CNL_ATTR_MY_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("my_size attr", mySizePtr, return ge::GRAPH_FAILED);

    const int64_t* lengthsSizePtr = attrs->GetAttrPointer<int64_t>(CNL_ATTR_LENGTHS_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("lengths_size attr", lengthsSizePtr, return ge::GRAPH_FAILED);

    auto* newLengthsShape = context->GetOutputShape(CNL_OUTPUT_NEW_LENGTHS_INDEX);
    OPS_LOG_E_IF_NULL("newLengthsShape", newLengthsShape, return ge::GRAPH_FAILED);

    newLengthsShape->SetDimNum(EXPECTED_RANK);
    newLengthsShape->SetDim(0, (*lengthsSizePtr) * (*mySizePtr));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeNewLengthsInferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto offsetsDtype = context->GetInputDataType(CNL_INPUT_OFFSETS_INDEX);
    context->SetOutputDataType(CNL_OUTPUT_NEW_LENGTHS_INDEX, offsetsDtype);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterNewIndicesInferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    const auto* indicesShape = context->GetInputShape(SNI_INPUT_INDICES_INDEX);
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const bool* sequencePtr = attrs->GetAttrPointer<bool>(SNI_ATTR_SEQUENCE_INDEX);
    OPS_LOG_E_IF_NULL("sequence attr", sequencePtr, return ge::GRAPH_FAILED);
    const bool* bucketizePosPtr = attrs->GetAttrPointer<bool>(SNI_ATTR_BUCKETIZE_POS_INDEX);
    OPS_LOG_E_IF_NULL("bucketize_pos attr", bucketizePosPtr, return ge::GRAPH_FAILED);

    const int64_t indicesSize = indicesShape->GetDim(0);
    const auto* weightsShape = context->GetInputShape(SNI_INPUT_WEIGHTS_INDEX);
    const bool hasWeights = (weightsShape != nullptr);

    auto* newIndicesShape = context->GetOutputShape(SNI_OUTPUT_NEW_INDICES_INDEX);
    auto* newWeightsShape = context->GetOutputShape(SNI_OUTPUT_NEW_WEIGHTS_INDEX);
    auto* newPosShape = context->GetOutputShape(SNI_OUTPUT_NEW_POS_INDEX);
    auto* unbucketizeShape = context->GetOutputShape(SNI_OUTPUT_UNBUCKETIZE_PERMUTE_INDEX);
    OPS_LOG_E_IF_NULL("newIndicesShape", newIndicesShape, return ge::GRAPH_FAILED);

    newIndicesShape->SetDimNum(EXPECTED_RANK);
    newIndicesShape->SetDim(0, indicesSize);
    if (newWeightsShape != nullptr) {
        newWeightsShape->SetDimNum(EXPECTED_RANK);
        newWeightsShape->SetDim(0, hasWeights ? indicesSize : 0);
    }
    if (newPosShape != nullptr) {
        newPosShape->SetDimNum(EXPECTED_RANK);
        newPosShape->SetDim(0, (*bucketizePosPtr) ? indicesSize : 0);
    }
    if (unbucketizeShape != nullptr) {
        unbucketizeShape->SetDimNum(EXPECTED_RANK);
        unbucketizeShape->SetDim(0, (*sequencePtr) ? indicesSize : 0);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScatterNewIndicesInferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    auto indicesDtype = context->GetInputDataType(SNI_INPUT_INDICES_INDEX);
    auto weightsDtype = context->GetInputDataType(SNI_INPUT_WEIGHTS_INDEX);

    context->SetOutputDataType(SNI_OUTPUT_NEW_INDICES_INDEX, indicesDtype);
    context->SetOutputDataType(SNI_OUTPUT_NEW_WEIGHTS_INDEX, weightsDtype);
    context->SetOutputDataType(SNI_OUTPUT_NEW_POS_INDEX, indicesDtype);
    context->SetOutputDataType(SNI_OUTPUT_UNBUCKETIZE_PERMUTE_INDEX, indicesDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class BlockBucketizeSparseFeaturesComputeNewLengths : public OpDef {
public:
    explicit BlockBucketizeSparseFeaturesComputeNewLengths(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("block_sizes")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("total_num_blocks")
            .ParamType(OPTIONAL)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("batch_size_offsets")
            .ParamType(OPTIONAL)
            .Follow("offsets", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("block_bucketize_pos_list")
            .ParamType(DYNAMIC)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Output("new_lengths")
            .ParamType(REQUIRED)
            .Follow("offsets", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Attr("my_size").Int();
        this->Attr("bucketize_pos").AttrType(OPTIONAL).Bool(false);
        this->Attr("lengths_size").Int();
        this->Attr("batch_size").Int();
        this->Attr("max_B").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::ComputeNewLengthsInferShape)
            .SetInferDataType(ge::ComputeNewLengthsInferDataType);
        this->AICore().SetTiling(optiling::ComputeNewLengthsTilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

class BlockBucketizeSparseFeaturesScatterNewIndices : public OpDef {
public:
    explicit BlockBucketizeSparseFeaturesScatterNewIndices(const char* name) : OpDef(name)
    {
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("block_sizes")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("new_offsets")
            .ParamType(REQUIRED)
            .Follow("offsets", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("total_num_blocks")
            .ParamType(OPTIONAL)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("batch_size_offsets")
            .ParamType(OPTIONAL)
            .Follow("offsets", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("block_bucketize_pos_list")
            .ParamType(DYNAMIC)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Output("new_indices")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("new_weights")
            .ParamType(OPTIONAL)
            .Follow("weights", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("new_pos")
            .ParamType(OPTIONAL)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Output("unbucketize_permute")
            .ParamType(OPTIONAL)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Attr("my_size").Int();
        this->Attr("bucketize_pos").AttrType(OPTIONAL).Bool(false);
        this->Attr("sequence").AttrType(OPTIONAL).Bool(false);
        this->Attr("keep_orig_idx").AttrType(OPTIONAL).Bool(false);
        this->Attr("lengths_size").Int();
        this->Attr("batch_size").Int();
        this->Attr("max_B").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::ScatterNewIndicesInferShape)
            .SetInferDataType(ge::ScatterNewIndicesInferDataType);
        this->AICore().SetTiling(optiling::ScatterNewIndicesTilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(BlockBucketizeSparseFeaturesComputeNewLengths);
OP_ADD(BlockBucketizeSparseFeaturesScatterNewIndices);
} // namespace ops
