
#include "embedding_lookup_by_address_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{

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
        TilingData1 tiling;

        size_t usrSize = 256;
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        if (CheckNullPointer(context, "Tiling context") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (CheckNullPointer(currentWorkspace, "currentWorkspace") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        currentWorkspace[0] = sysWorkspaceSize + usrSize;

        int32_t blockTotalNums = 48;
        int32_t ubLimit = 175 * 1024;
        auto *attrs = context->GetAttrs();
        if (CheckNullPointer(attrs, "GetAttrs attrs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const auto *attr0Value = attrs->GetAttrPointer<int64_t>(0);
        if (CheckNullPointer(attr0Value, " Lookup embbedingType attr0Value") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t embbedingDim = *attr0Value;
        if (embbedingDim <= 0) {
            printf("embbedingDim must larger than 0\n");
            return ge::GRAPH_FAILED;
        }

        const auto *attr1Value = attrs->GetAttrPointer<int64_t>(1);
        if (CheckNullPointer(attr1Value, "Lookup embbedingType attr1Value") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t embbedingType = *attr1Value;

        auto inputTensor = context->GetInputTensor(0);
        if (CheckNullPointer(inputTensor, "inputTensor") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t inputShape = inputTensor->GetShapeSize();
        int32_t singleDataSize = 4;
        if (embbedingType == 2) {
            singleDataSize = 2;
        }
        int32_t minMoveNum = 32 / singleDataSize;

        // onceMoveNums，(embbedingDim - 1 + minMoveNum) / min_move_num表示除以min_move_num向下取整
        int32_t onceMoveNums = minMoveNum * ((embbedingDim - 1 + minMoveNum) / minMoveNum);

        int32_t numToMove = (embbedingDim - 1 + onceMoveNums) / onceMoveNums;
        // 每个地址需要占用sizeof(int64_t)个字节，singleDataSize表示每个数据的字节数，需要使用2倍的内存空间，因为每次移动都需要复制一份数据
        int32_t pingPongNum = 1;
        int32_t occupyAddressBytesNum =
                sizeof(int64_t) + singleDataSize * onceMoveNums * numToMove * pingPongNum * 2;
        // 计算一轮计算中最多计算多少个addr，最后的 /4 再*4 是为了与32对齐，因为sizeof(int64_t) = 8
        int32_t addrMaxNum = (((ubLimit / occupyAddressBytesNum) / 4)) * 4;
        if (addrMaxNum <= 0) {
            return ge::GRAPH_FAILED;
        }

        tiling.set_embbeding_type(embbedingType);
        tiling.set_update_dim(embbedingDim);
        tiling.set_addr_nums(inputShape);
        tiling.set_ub_limit(ubLimit);

        tiling.set_addr_max_num(addrMaxNum);
        tiling.set_ping_pong_num(pingPongNum);
        tiling.set_single_data_size(singleDataSize);
        tiling.set_once_move_nums(onceMoveNums);

        context->SetBlockDim(blockTotalNums);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape1(gert::InferShapeContext *context)
    {

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
