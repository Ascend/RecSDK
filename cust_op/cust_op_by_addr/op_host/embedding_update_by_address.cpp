
#include "embedding_update_by_address_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
    struct TilingCompileInfo
    {
        int64_t ub_size;
    };

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TilingData2 tiling;

        int32_t block_total_nums = 48;
        int32_t ub_limit = 160 * 1024;

        int32_t update_dim;
        int32_t embbeding_type;

        int32_t input_shape = context->GetInputTensor(0)->GetShapeSize();
        int32_t input_dim = context->GetInputTensor(1)->GetShapeSize() / input_shape;
        int32_t update_type=*(context->GetAttrs()->GetAttrPointer<int64_t>(0));
        ge::DataType input_datatype = context->GetInputTensor(1)->GetDataType();

        switch (input_datatype)
        {
        case ge::DT_FLOAT16:
            embbeding_type = 2;
            break;
        case ge::DT_FLOAT:
            embbeding_type = 1;
            break;
        case ge::DT_INT32:
            embbeding_type = 0;
            break;
        default:
            embbeding_type = 1;
            break;
        }

        update_dim = input_dim;
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

    static ge::graphStatus TilingPrepare(gert::TilingParseContext *context)
    {
        return ge::GRAPH_SUCCESS;
    }

    static int check_op_support(const ge::Operator &op, ge::AscendString &result)
    {
        std::string res_json_str = "{\"ret_code\": \"0\",\"reason\": \"check_supported_stub\"}";
        result = ge::AscendString(res_json_str.c_str());
        return 1;
    }
}

namespace ge
{
    ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        gert::Shape *y_shape = context->GetOutputShape(0);
        int64_t input_shape = context->GetInputTensor(0)->GetShapeSize();
        int64_t input_dim = context->GetInputTensor(1)->GetShapeSize() / input_shape;
        y_shape->SetDimNum(2);
        y_shape->SetDim(0, input_shape);
        y_shape->SetDim(1, input_dim);
        return GRAPH_SUCCESS;
    }
    ge::graphStatus InferShapeRange(gert::InferShapeRangeContext *context)
    {
        return GRAPH_SUCCESS;
    }
    ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
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
                .SetTiling(optiling::TilingFunc)
                .SetTilingParse(optiling::TilingPrepare)
                .SetCheckSupport(optiling::check_op_support);

            OpAICoreConfig aicConfig;
            aicConfig.AsyncFlag(true)
                .DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(false)
                .RangeLimitValue("limited");
            this->AICore().AddConfig("ascend910b", aicConfig);
            this->AICore().AddConfig("ascend910", aicConfig);
        }
    };
    OP_ADD(EmbeddingUpdateByAddress, optiling::TilingCompileInfo);
}
