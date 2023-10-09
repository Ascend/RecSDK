
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
        if (CheckPointer(context, "Update embbeding_type context") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (CheckPointer(currentWorkspace, "currentWorkspace") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        currentWorkspace[0] = sysWorkspaceSize + usrSize;

        int32_t block_total_nums = 48;
        int32_t ub_limit = 175 * 1024;
        int32_t update_dim, embbeding_type;
        auto inputTensor = context->GetInputTensor(0);
        if (CheckPointer(inputTensor, "GetInputTensor inputTensor") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        int32_t input_shape = inputTensor->GetShapeSize();
        if (CheckPositiveInt(input_shape, "input_shape") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        auto inputTensor1 = context->GetInputTensor(1);
        if (CheckPointer(inputTensor1, "GetInputTensor inputTensor1") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        int32_t input_dim = inputTensor1->GetShapeSize() / input_shape;
        auto attrs = context->GetAttrs();
        if (CheckPointer(attrs, "GetAttrs attrs") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        auto attrPointer = attrs->GetAttrPointer<int64_t>(0);
        if (CheckPointer(attrPointer, "attrPointer") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        int32_t update_type = *(attrPointer);
        ge::DataType input_datatype = inputTensor1->GetDataType();
        if (input_datatype == ge::DT_FLOAT16) {
            embbeding_type = 2;
        } else if (input_datatype == ge::DT_INT32) {
            embbeding_type = 0;
        } else {
            embbeding_type = 1;
        }

        update_dim = input_dim;
        if (CheckPositiveInt(update_dim, "update_dim") != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;

        tiling.set_update_type(update_type);
        tiling.set_embbeding_type(embbeding_type);
        tiling.set_update_dim(update_dim);
        tiling.set_addr_nums(input_shape);
        tiling.set_ub_limit(ub_limit);
        context->SetBlockDim(block_total_nums);
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
