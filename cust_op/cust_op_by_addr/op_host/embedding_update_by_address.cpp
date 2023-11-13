
#include "embedding_update_by_address_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{

    template<typename T>
    static ge::graphStatus CheckPointer(T *pointer, const char *errorMessage)
    {
        if (pointer == nullptr) {
            printf("%s nullptr\n", errorMessage);
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus CheckPositiveInt(int32_t value, const char *errorMessage)
    {
        if (value <= 0) {
            printf("%s must larger than 0\n", errorMessage);
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TilingData2 tiling;

        size_t usrSize = 256, sysWorkspaceSize = 16 * 1024 * 1024;
        if (CheckPointer(context, "Update embbedingType context") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (CheckPointer(currentWorkspace, "currentWorkspace") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        currentWorkspace[0] = sysWorkspaceSize + usrSize;

        int32_t blockTotalNums = 48;
        int32_t ubLimit = 175 * 1024;

        auto inputTensor = context->GetInputTensor(0);
        if (CheckPointer(inputTensor, "GetInputTensor inputTensor") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        int32_t inputShape = inputTensor->GetShapeSize();
        if (CheckPositiveInt(inputShape, "inputShape") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        auto inputTensor1 = context->GetInputTensor(1);
        if (CheckPointer(inputTensor1, "GetInputTensor inputTensor1") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        int32_t inputDim = inputTensor1->GetShapeSize() / inputShape;
        if (CheckPositiveInt(inputDim, "inputDim") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto attrs = context->GetAttrs();
        if (CheckPointer(attrs, "GetAttrs attrs") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        auto attrPointer = attrs->GetAttrPointer<int64_t>(0);
        if (CheckPointer(attrPointer, "attrPointer") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        int32_t updateType = *(attrPointer);
        ge::DataType inputDatatype = inputTensor1->GetDataType();
        int32_t embbedingType;
        if (inputDatatype == ge::DT_FLOAT16) {
            embbedingType = 2;
        } else if (inputDatatype == ge::DT_INT32) {
            embbedingType = 0;
        } else {
            embbedingType = 1;
        }

        int32_t singleDataSize = 4;
        if (embbedingType == 2) {
            singleDataSize = 2;
        }
        int32_t minMoveNum = 32 / singleDataSize;

        // onceMoveNums，(updateDim - 1 + minMoveNum) / min_move_num表示除以min_move_num向下取整
        int32_t onceMoveNums = minMoveNum * ((inputDim - 1 + minMoveNum) / minMoveNum);

        int32_t numToMove = (inputDim - 1 + onceMoveNums) / onceMoveNums;
        // 每个地址需要占用sizeof(int64_t)个字节，singleDataSize表示每个数据的字节数，需要使用2倍的内存空间，因为每次移动都需要复制一份数据
        int32_t pingPongNum = 1;
        int32_t occupyAddressBytesNum =
                sizeof(int64_t) + singleDataSize * onceMoveNums * numToMove * pingPongNum * 2;
        // 计算一轮计算中最多计算多少个addr，最后的 /4 再*4 是为了与32对齐，因为sizeof(int64_t) = 8
        int32_t addrMaxNum = ((int)((int)(ubLimit / occupyAddressBytesNum) / 4)) * 4;
        if (CheckPositiveInt(addrMaxNum, "addrMaxNum") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        tiling.set_update_type(updateType);
        tiling.set_embbeding_type(embbedingType);
        tiling.set_update_dim(inputDim);
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
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, ge::DataType(DT_FLOAT));
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class EmbeddingUpdateByAddress : public OpDef
    {
    public:
        EmbeddingUpdateByAddress(const char *name) : OpDef(name)
        {
            this->Input("address")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("embedding")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("update_type").AttrType(OPTIONAL).Int(0);
            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

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
    OP_ADD(EmbeddingUpdateByAddress);
}
