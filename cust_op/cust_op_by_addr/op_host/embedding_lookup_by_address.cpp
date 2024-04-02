/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "embedding_lookup_by_address_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
    constexpr int32_t BLOCK_DIM = 48;  // 910b一张卡48个vector核
    constexpr int32_t SIZE_OF_HALF = 2;
    constexpr int32_t SIZE_OF_FLOAT_OR_INT = 4;
    constexpr int32_t MIN_BLOCK_SIZE = 32; // ub空间的数据都要按照32对齐
    constexpr int32_t UB_LIMIT = 175 * 1024;
    constexpr int32_t USR_SIZE = 256;
    constexpr int32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
    constexpr int32_t PING_PONG_NUM = 1;

    template <typename T>
    static ge::graphStatus CheckNullPointer(T *pointer, const char *errorMessage)
    {
        if (pointer == nullptr) {
            printf("%s nullptr\n", errorMessage);
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        if (CheckNullPointer(context, "Tiling context") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 设备侧Global Memory上的一块内存
        if (CheckNullPointer(currentWorkspace, "currentWorkspace") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = SYS_WORKSPACE_SIZE + USR_SIZE;

        auto *attrs = context->GetAttrs();
        if (CheckNullPointer(attrs, "GetAttrs attrs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const auto *attr0Value = attrs->GetAttrPointer<int64_t>(0);
        if (CheckNullPointer(attr0Value, " Lookup embbedingType attr0Value") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int32_t embeddingDim = *attr0Value;
        if (embeddingDim <= 0) {
            printf("embeddingDim must larger than 0\n");
            return ge::GRAPH_FAILED;
        }

        const auto *attr1Value = attrs->GetAttrPointer<int64_t>(1);
        if (CheckNullPointer(attr1Value, "Lookup embbedingType attr1Value") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int32_t embeddingType = *attr1Value; // 0:int32; 1:float; 2:half
        if (embeddingType > 2 || embeddingType < 0) {
            return ge::GRAPH_FAILED;
        }

        auto inputTensor = context->GetInputTensor(0);
        if (CheckNullPointer(inputTensor, "inputTensor") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t inputShape = inputTensor->GetShapeSize();

        int32_t typeSize = SIZE_OF_FLOAT_OR_INT;
        if (embeddingType == 2) {
            typeSize = SIZE_OF_HALF;
        }
        // shape需要对齐到的最小单位, MIN_BLOCK_SIZE=32
        int32_t alignNum = MIN_BLOCK_SIZE / typeSize;
        // embeddingDimAligned，表示需要向上对齐到最小单位
        int32_t embeddingDimAligned = ((embeddingDim - 1 + alignNum) / alignNum) * alignNum;
        // LocalTensor空间，tbuf存储int64的地址+inQueue+outQueue两倍的emb，因为每次移动都需要复制一份数据
        int32_t occupyAddressBytesNum =
                sizeof(int64_t) + typeSize * embeddingDimAligned * PING_PONG_NUM * 2;
        // 一轮计算中最多计算多少个addr，由于地址也要搬到ub，所以需要对齐32,
        int32_t addrPerLoop = (UB_LIMIT / occupyAddressBytesNum) & (~3); //  & (~3)，保证地址数是4的倍数
        if (addrPerLoop <= 0) {
            return ge::GRAPH_FAILED;
        }
        TilingData1 tiling;
        tiling.set_ping_pong_num(PING_PONG_NUM);
        tiling.set_addr_nums(inputShape);
        tiling.set_embedding_type(embeddingType);
        tiling.set_embedding_dim(embeddingDim);

        tiling.set_addr_per_loop(addrPerLoop);
        tiling.set_type_size(typeSize);
        tiling.set_emb_dim_aligned(embeddingDimAligned);

        // 和tiling set 区别开， BlockDim就是BlockNum，可以理解为卡的核数
        context->SetBlockDim(BLOCK_DIM);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape1(gert::InferShapeContext *context)
    {

        if (optiling::CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        gert::Shape *yShape = context->GetOutputShape(0);
        if (optiling::CheckNullPointer(yShape, "yShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto *attrs = context->GetAttrs();
        if (optiling::CheckNullPointer(attrs, "attrs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        const auto *attr0Value = attrs->GetAttrPointer<int64_t>(0);
        if (optiling::CheckNullPointer(attr0Value, "Lookup embbedingType attr0Value") != ge::GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }

        int64_t updateDim = *attr0Value;

        int64_t inputShape = context->GetInputTensor(0)->GetShapeSize();
        yShape->SetDimNum(2);
        yShape->SetDim(0, inputShape);
        yShape->SetDim(1, updateDim);
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType1(gert::InferDataTypeContext *context)
    {

        int64_t embbedingType;
        if (optiling::CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto *attrs = context->GetAttrs();
        if (optiling::CheckNullPointer(attrs, "attrs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        const auto *attr1Value = attrs->GetAttrPointer<int64_t>(1);
        if (optiling::CheckNullPointer(attr1Value, "Lookup embbedingType") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        embbedingType = *attr1Value;
        if (embbedingType == 0)
        {
            context->SetOutputDataType(0, ge::DataType(DT_INT32));
        }
        else if (embbedingType == 1)
        {
            context->SetOutputDataType(0, ge::DataType(DT_FLOAT));
        }
        else if (embbedingType == 2)
        {

            context->SetOutputDataType(0, ge::DataType(DT_FLOAT16));
        }
        else
        {
            context->SetOutputDataType(0, ge::DataType(DT_FLOAT));
        }

        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class EmbeddingLookupByAddress : public OpDef
    {
    public:
        EmbeddingLookupByAddress(const char *name) : OpDef(name)
        {
            this->Input("address")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("embedding_dim").AttrType(OPTIONAL).Int(32);
            this->Attr("embedding_type").AttrType(OPTIONAL).Int(1);

            this->SetInferShape(ge::InferShape1)
                .SetInferDataType(ge::InferDataType1);

            this->AICore()
                .SetTiling(optiling::TilingFunc);

            OpAICoreConfig aicConfig;
            aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(false);

            this->AICore().AddConfig("ascend910b", aicConfig);
            this->AICore().AddConfig("ascend910", aicConfig);
        }
    };

    OP_ADD(EmbeddingLookupByAddress);
}
