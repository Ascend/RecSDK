
#include "embedding_update_by_address_tiling.h"
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
        if (CheckPointer(context, "Update embeddingType context") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (CheckPointer(currentWorkspace, "currentWorkspace") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = SYS_WORKSPACE_SIZE + USR_SIZE;

        auto inputTensor = context->GetInputTensor(0);
        if (CheckPointer(inputTensor, "GetInputTensor inputTensor") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t inputShape = inputTensor->GetShapeSize();
        if (CheckPositiveInt(inputShape, "inputShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto inputTensor1 = context->GetInputTensor(1);
        if (CheckPointer(inputTensor1, "GetInputTensor inputTensor1") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t inputDim = inputTensor1->GetShapeSize() / inputShape;
        if (CheckPositiveInt(inputDim, "inputDim") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto attrs = context->GetAttrs();
        if (CheckPointer(attrs, "GetAttrs attrs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        auto attrPointer = attrs->GetAttrPointer<int64_t>(0);
        if (CheckPointer(attrPointer, "attrPointer") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        int32_t updateType = *(attrPointer);
        ge::DataType inputDatatype = inputTensor1->GetDataType();
        int32_t embeddingType;
        if (inputDatatype == ge::DT_FLOAT16) {
            embeddingType = 2;
        } else if (inputDatatype == ge::DT_INT32) {
            embeddingType = 0;
        } else {
            embeddingType = 1;
        }

        int32_t typeSize = SIZE_OF_FLOAT_OR_INT;
        if (embeddingType == 2) {
            typeSize = SIZE_OF_HALF;
        }
        int32_t alignNum = MIN_BLOCK_SIZE / typeSize;

        int32_t inputDimAligned = alignNum * ((inputDim - 1 + alignNum) / alignNum);

        // 每个地址需要占用sizeof(int64_t)个字节，singleDataSize表示每个数据的字节数，需要使用2倍的内存空间，因为每次移动都需要复制一份数据
        int32_t occupyAddressBytesNum =
                sizeof(int64_t) + typeSize * inputDimAligned * PING_PONG_NUM * 2;
        // 一轮计算中最多计算多少个addr，由于地址也要搬到ub，所以需要对齐32
        int32_t addrPerLoop = (UB_LIMIT / occupyAddressBytesNum) & (~3); // & (~3)，保证地址数是4的倍数
        if (CheckPositiveInt(addrPerLoop, "addrPerLoop") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        TilingData2 tiling;
        tiling.set_ping_pong_num(PING_PONG_NUM);

        tiling.set_update_type(updateType);
        tiling.set_embedding_type(embeddingType);
        tiling.set_update_dim(inputDim);
        tiling.set_addr_nums(inputShape);
        tiling.set_addr_per_loop(addrPerLoop);
        tiling.set_type_size(typeSize);
        tiling.set_input_dim_aligned(inputDimAligned);

        context->SetBlockDim(BLOCK_DIM);
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
