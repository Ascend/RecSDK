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

enum InputIndex : int32_t {
    INPUT_LENGTHS_INDEX = 0,
    INPUT_INDICES_INDEX = 1,
    INPUT_BLOCK_SIZES_INDEX = 2,
    INPUT_WEIGHTS_INDEX = 3,
    INPUT_BATCH_SIZE_PER_FEATURE_INDEX = 4,
    INPUT_TOTAL_NUM_BLOCKS_INDEX = 5,
    INPUT_BLOCK_BUCKETIZE_POS_INDEX = 6,
};

enum OutputIndex : int32_t {
    OUTPUT_NEW_LENGTHS_INDEX = 0,
    OUTPUT_NEW_INDICES_INDEX = 1,
    OUTPUT_NEW_WEIGHTS_INDEX = 2,
    OUTPUT_NEW_POS_INDEX = 3,
    OUTPUT_UNBUCKETIZE_PERMUTE_INDEX = 4,
};

enum AttrIndex : int32_t {
    ATTR_MY_SIZE_INDEX = 0,
    ATTR_BUCKETIZE_POS_INDEX = 1,
    ATTR_SEQUENCE_INDEX = 2,
    ATTR_KEEP_ORIG_IDX_INDEX = 3,
    ATTR_MAX_B_INDEX = 4,
};

constexpr int32_t EXPECTED_RANK = 1;
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t SMALL_DATA_THRESHOLD_INT32 = 24 * MAX_THREADS_PER_BLOCK;
constexpr int32_t SMALL_DATA_THRESHOLD_INT64 = 44 * MAX_THREADS_PER_BLOCK;
constexpr uint32_t CACHE_ALIGN = 64;
constexpr size_t LOCAL_MEMORY_SIZE = 216 * 1024;

/* 快除法预计算（参考 HierarchicalKV precomputation_for_kernel_div）结果通过 tiling 传给 kernel */
inline void HostPrecomputeFastDivmod64(uint64_t divisor, uint64_t& outMagic, uint32_t& outShift)
{
    if (divisor <= 1) {
        outMagic = 0;
        outShift = 0;
        return;
    }
    unsigned __int128 one = 1;
    uint32_t s = 0;
    for (; s < 64; ++s) {
        if ((one << s) >= static_cast<unsigned __int128>(divisor)) {
            break;
        }
    }
    outShift = s;
    outMagic = static_cast<uint64_t>(
        ((one << 64) * ((one << s) - static_cast<unsigned __int128>(divisor))) /
        static_cast<unsigned __int128>(divisor) + 1);
}

inline int64_t ComputeTotalBlocks(int64_t totalLength, bool isInt32)
{
    if (totalLength <= 0) {
        return 0;
    }
    const int64_t smallThreshold = isInt32 ? SMALL_DATA_THRESHOLD_INT32 : SMALL_DATA_THRESHOLD_INT64;
    const int64_t perBlockCapacity = (totalLength <= smallThreshold)
        ? static_cast<int64_t>(MAX_THREADS_PER_BLOCK)
        : static_cast<int64_t>(MAX_THREADS_PER_BLOCK) * MAX_ELEMENTS_PER_THREAD;
    return (totalLength + perBlockCapacity - 1) / perBlockCapacity;
}

} // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("lengthsShape", context->GetInputShape(INPUT_LENGTHS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesShape", context->GetInputShape(INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("blockSizesShape", context->GetInputShape(INPUT_BLOCK_SIZES_INDEX), return ge::GRAPH_FAILED);
    const auto* totalNumBlocksShape = context->GetInputShape(INPUT_TOTAL_NUM_BLOCKS_INDEX);
    const auto* batchSizePerFeatureShape = context->GetInputShape(INPUT_BATCH_SIZE_PER_FEATURE_INDEX);
    const auto* posTensorShape = context->GetInputShape(INPUT_BLOCK_BUCKETIZE_POS_INDEX);
    OPS_LOG_E_IF_NULL("lengthsTensor", context->GetInputTensor(INPUT_LENGTHS_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesTensor", context->GetInputTensor(INPUT_INDICES_INDEX), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("blockSizesTensor", context->GetInputTensor(INPUT_BLOCK_SIZES_INDEX), return ge::GRAPH_FAILED);

    const auto* lengthsShape = context->GetInputShape(INPUT_LENGTHS_INDEX);
    const auto* indicesShape = context->GetInputShape(INPUT_INDICES_INDEX);
    const auto* blockSizesShape = context->GetInputShape(INPUT_BLOCK_SIZES_INDEX);
    const auto* batchSizePerFeatureTensor = context->GetInputTensor(INPUT_BATCH_SIZE_PER_FEATURE_INDEX);
    const auto lengthsType = context->GetInputTensor(INPUT_LENGTHS_INDEX)->GetDataType();
    const auto indicesType = context->GetInputTensor(INPUT_INDICES_INDEX)->GetDataType();
    const auto blockSizesType = context->GetInputTensor(INPUT_BLOCK_SIZES_INDEX)->GetDataType();
    const auto lengthsStorageShape = lengthsShape->GetStorageShape();
    const auto indicesStorageShape = indicesShape->GetStorageShape();
    const auto blockSizesStorageShape = blockSizesShape->GetStorageShape();
    const bool enableBatchSizePerFeature = (batchSizePerFeatureShape != nullptr);

    OPS_CHECK(lengthsStorageShape.GetDimNum() != EXPECTED_RANK,
        OPS_LOG_E("[ERROR]", "lengths must be 1D"), return ge::GRAPH_FAILED);
    OPS_CHECK(indicesStorageShape.GetDimNum() != EXPECTED_RANK,
        OPS_LOG_E("[ERROR]", "indices must be 1D"), return ge::GRAPH_FAILED);
    OPS_CHECK(blockSizesStorageShape.GetDimNum() != EXPECTED_RANK,
        OPS_LOG_E("[ERROR]", "block_sizes must be 1D"), return ge::GRAPH_FAILED);

    const int64_t lengthsSize = lengthsStorageShape.GetShapeSize();
    const int64_t indicesSize = indicesStorageShape.GetShapeSize();
    const int64_t numFeatures = blockSizesStorageShape.GetShapeSize();
    OPS_CHECK(lengthsSize <= 0 || indicesSize < 0 || numFeatures <= 0,
        OPS_LOG_E("[ERROR]", "Invalid tensor shapes for block bucketize"), return ge::GRAPH_FAILED);
    OPS_CHECK(!enableBatchSizePerFeature && (lengthsSize % numFeatures != 0),
        OPS_LOG_E("[ERROR]", "lengths size must be divisible by block_sizes size"), return ge::GRAPH_FAILED);

    auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    const int64_t* mySizePtr = attrs->GetAttrPointer<int64_t>(ATTR_MY_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("my_size attr", mySizePtr, return ge::GRAPH_FAILED);
    const int64_t mySize = *mySizePtr;
    OPS_CHECK(mySize <= 0,
        OPS_LOG_E("[ERROR]", "my_size must be positive"), return ge::GRAPH_FAILED);

    const bool* bucketizePosPtr = attrs->GetAttrPointer<bool>(ATTR_BUCKETIZE_POS_INDEX);
    OPS_LOG_E_IF_NULL("bucketize_pos attr", bucketizePosPtr, return ge::GRAPH_FAILED);

    const bool* sequencePtr = attrs->GetAttrPointer<bool>(ATTR_SEQUENCE_INDEX);
    OPS_LOG_E_IF_NULL("sequence attr", sequencePtr, return ge::GRAPH_FAILED);

    const bool* keepOrigIdxPtr = attrs->GetAttrPointer<bool>(ATTR_KEEP_ORIG_IDX_INDEX);
    OPS_LOG_E_IF_NULL("keep_orig_idx attr", keepOrigIdxPtr, return ge::GRAPH_FAILED);

    const int64_t* maxBPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_B_INDEX);
    OPS_LOG_E_IF_NULL("max_B attr", maxBPtr, return ge::GRAPH_FAILED);
    const int64_t maxB = *maxBPtr;
    OPS_CHECK(enableBatchSizePerFeature && maxB <= 0,
        OPS_LOG_E("[ERROR]", "max_B must be positive when batch_size_per_feature is provided"),
            return ge::GRAPH_FAILED);

    OPS_CHECK(blockSizesType != indicesType,
        OPS_LOG_E("[ERROR]", "block_sizes dtype must match indices"), return ge::GRAPH_FAILED);
    if (enableBatchSizePerFeature) {
        OPS_LOG_E_IF_NULL("batch_size_per_feature Tensor", batchSizePerFeatureTensor, return ge::GRAPH_FAILED);
        OPS_CHECK(batchSizePerFeatureTensor->GetDataType() != lengthsType,
            OPS_LOG_E("[ERROR]", "batch_size_per_feature dtype must match lengths"), return ge::GRAPH_FAILED);
    }

    const int64_t batchSize = enableBatchSizePerFeature ? 0 : (lengthsSize / numFeatures);
    const int64_t newLengthsSize = lengthsSize * mySize;

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspace", workspace, return ge::GRAPH_FAILED);
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    const bool isOffsetInt32 = (lengthsType == ge::DT_INT32);
    const bool isIndexInt32 = (indicesType == ge::DT_INT32);
    const uint32_t offsetElementSize = isOffsetInt32 ? sizeof(int32_t) : sizeof(int64_t);
    const uint32_t indexElementSize = isIndexInt32 ? sizeof(int32_t) : sizeof(int64_t);
    bool hasBucketizePosList = false;
    if (posTensorShape != nullptr) {
        const auto posStorageShape = posTensorShape->GetStorageShape();
        hasBucketizePosList = (posStorageShape.GetShapeSize() > 0);
    }
    constexpr uint64_t ALIGN64 = 64;

    const uint64_t offsetsSize = static_cast<uint64_t>(lengthsSize + 1) * offsetElementSize;
    const uint64_t newOffsetsSize = static_cast<uint64_t>(newLengthsSize + 1) * offsetElementSize;
    const uint64_t writeOffsetsSize = static_cast<uint64_t>(newLengthsSize) * offsetElementSize;
    const uint64_t batchSizeOffsetsSize = enableBatchSizePerFeature ?
        static_cast<uint64_t>(numFeatures + 1) * offsetElementSize : 0;
    const uint64_t posPtrsSize = hasBucketizePosList ?
        static_cast<uint64_t>(numFeatures) * sizeof(uint64_t) : 0;
    const uint64_t posLensSize = hasBucketizePosList ?
        static_cast<uint64_t>(numFeatures) * sizeof(int64_t) : 0;

    const int64_t totalBlocksForLengths = ComputeTotalBlocks(lengthsSize, isOffsetInt32);
    const int64_t totalBlocksForBatchSize = ComputeTotalBlocks(numFeatures, isOffsetInt32);
    const int64_t totalBlocksForNewLengths = ComputeTotalBlocks(newLengthsSize, isOffsetInt32);

    // kernel 使用 512 threads / 32 warp_size = 16 warps，每个 warp 处理一个 row
    constexpr int64_t KERNEL_WARPS_PER_BLOCK = 16;
    const int64_t blocksForRows = (lengthsSize + KERNEL_WARPS_PER_BLOCK - 1) / KERNEL_WARPS_PER_BLOCK;

    int64_t totalBlocksMax =
        (totalBlocksForLengths >= totalBlocksForBatchSize && totalBlocksForLengths >= totalBlocksForNewLengths) ?
        totalBlocksForLengths : (totalBlocksForBatchSize >= totalBlocksForNewLengths ?
        totalBlocksForBatchSize : totalBlocksForNewLengths);
    
    if (blocksForRows > totalBlocksMax) {
        totalBlocksMax = blocksForRows;
    }

    if (totalBlocksMax <= 0) {
        totalBlocksMax = 1;
    }
    const uint64_t blockSumsBytes = static_cast<uint64_t>(totalBlocksMax) * CACHE_ALIGN;

    uint64_t nextOffset = 0;
    const uint64_t offsetsOffsetVal = nextOffset;
    nextOffset += offsetsSize;

    nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
    const uint64_t newOffsetsOffsetVal = nextOffset;
    nextOffset += newOffsetsSize;

    nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
    const uint64_t writeOffsetsOffsetVal = nextOffset;
    nextOffset += writeOffsetsSize;

    uint64_t batchSizeOffsetsOffsetVal = 0;
    if (enableBatchSizePerFeature) {
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        batchSizeOffsetsOffsetVal = nextOffset;
        nextOffset += batchSizeOffsetsSize;
    }

    uint64_t posPtrsOffsetVal = 0;
    uint64_t posLensOffsetVal = 0;
    if (hasBucketizePosList) {
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        posPtrsOffsetVal = nextOffset;
        nextOffset += posPtrsSize;
        nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
        posLensOffsetVal = nextOffset;
        nextOffset += posLensSize;
    }

    nextOffset = (nextOffset + ALIGN64 - 1) / ALIGN64 * ALIGN64;
    const uint64_t blockSumsOffsetVal = nextOffset;
    nextOffset += blockSumsBytes;

    const uint64_t userWorkspaceSize = nextOffset;
    workspace[0] = systemWorkspaceSize + userWorkspaceSize;

    const int64_t totalBlocks = totalBlocksMax;
    const size_t maxCores = ascendPlatform.GetCoreNumAiv();
    size_t coreNum = (totalBlocks <= 0) ? 1 : (totalBlocks < static_cast<int64_t>(maxCores) ?
        static_cast<size_t>(totalBlocks) : maxCores);
    coreNum = (coreNum == 0) ? 1 : coreNum;

    BlockBucketizeSparseFeaturesTilingData tiling;
    tiling.set_lengthsSize(lengthsSize);
    tiling.set_indicesSize(indicesSize);
    tiling.set_numFeatures(numFeatures);
    tiling.set_batchSize(batchSize);
    tiling.set_mySize(mySize);
    tiling.set_newLengthsSize(newLengthsSize);
    tiling.set_maxBatchSize(maxB);
    tiling.set_enableSequence(*sequencePtr);
    const auto* weightsTensor = context->GetInputTensor(INPUT_WEIGHTS_INDEX);
    tiling.set_enableWeights(weightsTensor != nullptr);
    tiling.set_enableBucketizePos(*bucketizePosPtr);
    tiling.set_enableKeepOrigIdx(*keepOrigIdxPtr);
    tiling.set_enableBatchSizePerFeature(enableBatchSizePerFeature);
    const bool enableTotalNumBlocks = (totalNumBlocksShape != nullptr);
    tiling.set_enableTotalNumBlocks(enableTotalNumBlocks);
    tiling.set_enableBlockBucketizePos(hasBucketizePosList);

    uint64_t mySizeMagic = 0;
    uint32_t mySizeShift = 0;
    HostPrecomputeFastDivmod64(static_cast<uint64_t>(mySize), mySizeMagic, mySizeShift);
    tiling.set_mySizeDivMagic(mySizeMagic);
    tiling.set_mySizeDivShift(mySizeShift);

    tiling.set_offsetsOffset(offsetsOffsetVal);
    tiling.set_newOffsetsOffset(newOffsetsOffsetVal);
    tiling.set_writeOffsetsOffset(writeOffsetsOffsetVal);
    tiling.set_batchSizeOffsetsOffset(batchSizeOffsetsOffsetVal);
    tiling.set_posPtrsOffset(posPtrsOffsetVal);
    tiling.set_posLensOffset(posLensOffsetVal);
    tiling.set_blockSumsOffset(blockSumsOffsetVal);
    if (enableTotalNumBlocks) {
        const auto totalNumBlocksStorageShape = totalNumBlocksShape->GetStorageShape();
        OPS_CHECK(totalNumBlocksStorageShape.GetDimNum() != EXPECTED_RANK,
            OPS_LOG_E("[ERROR]", "total_num_blocks must be 1D"), return ge::GRAPH_FAILED);
        OPS_CHECK(totalNumBlocksStorageShape.GetShapeSize() != numFeatures,
            OPS_LOG_E("[ERROR]", "total_num_blocks must match block_sizes length"), return ge::GRAPH_FAILED);
        const auto* totalNumBlocksTensor = context->GetInputTensor(INPUT_TOTAL_NUM_BLOCKS_INDEX);
        OPS_LOG_E_IF_NULL("total_num_blocks Tensor", totalNumBlocksTensor, return ge::GRAPH_FAILED);
        OPS_CHECK(totalNumBlocksTensor->GetDataType() != indicesType,
            OPS_LOG_E("[ERROR]", "total_num_blocks dtype must match indices"), return ge::GRAPH_FAILED);
    }
    if (enableBatchSizePerFeature) {
        const auto batchSizePerFeatureStorageShape = batchSizePerFeatureShape->GetStorageShape();
        OPS_CHECK(batchSizePerFeatureStorageShape.GetDimNum() != EXPECTED_RANK,
            OPS_LOG_E("[ERROR]", "batch_size_per_feature must be 1D"), return ge::GRAPH_FAILED);
        OPS_CHECK(batchSizePerFeatureStorageShape.GetShapeSize() != numFeatures,
            OPS_LOG_E("[ERROR]", "batch_size_per_feature length must match block_sizes"), return ge::GRAPH_FAILED);
    }

    context->SetBlockDim(coreNum);
    context->SetLocalMemorySize(LOCAL_MEMORY_SIZE);
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const auto* lengthsShape = context->GetInputShape(INPUT_LENGTHS_INDEX);
    const auto* indicesShape = context->GetInputShape(INPUT_INDICES_INDEX);
    const auto* weightsShape = context->GetInputShape(INPUT_WEIGHTS_INDEX);
    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("lengthsShape", lengthsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("indicesShape", indicesShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("attrs", attrs, return ge::GRAPH_FAILED);

    auto* newLengthsShape = context->GetOutputShape(OUTPUT_NEW_LENGTHS_INDEX);
    auto* newIndicesShape = context->GetOutputShape(OUTPUT_NEW_INDICES_INDEX);
    auto* newWeightsShape = context->GetOutputShape(OUTPUT_NEW_WEIGHTS_INDEX);
    auto* newPosShape = context->GetOutputShape(OUTPUT_NEW_POS_INDEX);
    auto* unbucketizeShape = context->GetOutputShape(OUTPUT_UNBUCKETIZE_PERMUTE_INDEX);
    OPS_LOG_E_IF_NULL("newLengthsShape", newLengthsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("newIndicesShape", newIndicesShape, return ge::GRAPH_FAILED);

    const int64_t* mySizePtr = attrs->GetAttrPointer<int64_t>(ATTR_MY_SIZE_INDEX);
    OPS_LOG_E_IF_NULL("my_size attr NULL", mySizePtr, return ge::GRAPH_FAILED);
    const int64_t mySize = *mySizePtr;
    const bool* sequencePtr = attrs->GetAttrPointer<bool>(ATTR_SEQUENCE_INDEX);
    OPS_LOG_E_IF_NULL("sequence attr NULL", sequencePtr, return ge::GRAPH_FAILED);
    const bool* bucketizePosPtr = attrs->GetAttrPointer<bool>(ATTR_BUCKETIZE_POS_INDEX);
    OPS_LOG_E_IF_NULL("bucketize_pos attr NULL", bucketizePosPtr, return ge::GRAPH_FAILED);
    const bool hasWeights = (weightsShape != nullptr);

    const int64_t lengthsSize = lengthsShape->GetDim(0);
    const int64_t indicesSize = indicesShape->GetDim(0);

    newLengthsShape->SetDimNum(EXPECTED_RANK);
    newLengthsShape->SetDim(0, lengthsSize * mySize);

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

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto lengthsDtype = context->GetInputDataType(INPUT_LENGTHS_INDEX);
    auto indicesDtype = context->GetInputDataType(INPUT_INDICES_INDEX);
    auto weightsDtype = context->GetInputDataType(INPUT_WEIGHTS_INDEX);

    context->SetOutputDataType(OUTPUT_NEW_LENGTHS_INDEX, lengthsDtype);
    context->SetOutputDataType(OUTPUT_NEW_INDICES_INDEX, indicesDtype);
    context->SetOutputDataType(OUTPUT_NEW_WEIGHTS_INDEX, weightsDtype);
    context->SetOutputDataType(OUTPUT_NEW_POS_INDEX, indicesDtype);
    context->SetOutputDataType(OUTPUT_UNBUCKETIZE_PERMUTE_INDEX, indicesDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class BlockBucketizeSparseFeatures : public OpDef {
public:
    explicit BlockBucketizeSparseFeatures(const char* name) : OpDef(name)
    {
        this->Input("lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("block_sizes")
            .ParamType(REQUIRED)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("batch_size_per_feature")
            .ParamType(OPTIONAL)
            .Follow("lengths", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("total_num_blocks")
            .ParamType(OPTIONAL)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("block_bucketize_pos_list")
            .ParamType(DYNAMIC)
            .Follow("indices", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});

        this->Output("new_lengths")
            .ParamType(REQUIRED)
            .Follow("lengths", FollowType::DTYPE)
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
        this->Attr("max_B").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(BlockBucketizeSparseFeatures);
} // namespace ops
