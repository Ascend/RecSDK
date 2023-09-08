
#include "embedding_lookup_by_address_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TilingData1 tiling;

        size_t usrSize = 256;
        size_t sysWorkspaceSize = 16 * 1024 * 1024;
        if (context == nullptr) {
            printf("Tiling context nullptr\n");
            return ge::GRAPH_FAILED;
        }

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            printf("currentWorkspace nullptr\n");
            return ge::GRAPH_FAILED;
        }

        currentWorkspace[0] = sysWorkspaceSize + usrSize;

        int32_t block_total_nums = 48;
        int32_t ub_limit = 175 * 1024;
        auto *attrs = context->GetAttrs();
        const auto *attr0_value = attrs->GetAttrPointer<int64_t>(0);
        if (attr0_value == nullptr) {
            printf(" Lookup embbeding_type attr0_value nullptr\n");
            return ge::GRAPH_FAILED;
        }

        int32_t embbeding_dim = *attr0_value;
        if (embbeding_dim <= 0) {
            printf("embbeding_dim must larger than 0\n");
            return ge::GRAPH_FAILED;
        }

        const auto *attr1_value = attrs->GetAttrPointer<int64_t>(1);
        if (attr1_value == nullptr) {
            printf(" Lookup embbeding_type attr1_value nullptr\n");
            return ge::GRAPH_FAILED;
        }

        int32_t embbeding_type = *attr1_value;

        int32_t input_shape = context->GetInputTensor(0)->GetShapeSize();

        tiling.set_embbeding_type(embbeding_type);
        tiling.set_update_dim(embbeding_dim);
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
    static ge::graphStatus InferShape1(gert::InferShapeContext *context)
    {

        gert::Shape *y_shape = context->GetOutputShape(0);
        auto *attrs = context->GetAttrs();
        const auto *attr0_value = attrs->GetAttrPointer<int64_t>(0);
        if (attr0_value == nullptr) {
            printf(" Lookup embbeding_type attr0_value nullptr\n");
            return GRAPH_FAILED;
        }

        int64_t update_dim = *attr0_value;

        int64_t input_shape = context->GetInputTensor(0)->GetShapeSize();
        y_shape->SetDimNum(2);
        y_shape->SetDim(0, input_shape);
        y_shape->SetDim(1, update_dim);
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType1(gert::InferDataTypeContext *context)
    {

        int64_t embbeding_type;
        auto *attrs = context->GetAttrs();
        const auto *attr1_value = attrs->GetAttrPointer<int64_t>(1);
        if (attr1_value == nullptr) {
            printf(" Lookup embbeding_type nullptr\n");
        }
        else
        {
            embbeding_type = *attr1_value;
        }
        if (embbeding_type == 0)
        {
            context->SetOutputDataType(0, ge::DataType(DT_INT32));
        }
        else if (embbeding_type == 1)
        {
            context->SetOutputDataType(0, ge::DataType(DT_FLOAT));
        }
        else if (embbeding_type == 2)
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
