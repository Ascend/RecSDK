/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <unistd.h>
#include "default_gather_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
uint32_t SizeOf(ge::DataType type)
{
    if (type == ge::DT_INT64) {
        return 8;
    }

    if (type == ge::DT_FLOAT || type == ge::DT_INT32) {
        return 4;
    }

    return 0;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DefaultGatherTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::Tensor* x1_tensor = context->GetInputTensor(0);
    ge::DataType idx_dtype = x1_tensor->GetDataType();
    uint32_t idx_size = SizeOf(idx_dtype);

    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    const gert::Tensor* x2_tensor = context->GetInputTensor(1);
    ge::DataType table_dtype = x2_tensor->GetDataType();
    uint32_t table_size = SizeOf(table_dtype);

    const gert::StorageShape* y_shape = context->GetOutputShape(0);
    uint64_t ubSize;
    uint32_t Idcolumn;
    uint32_t Tablerow;
    uint32_t Tablecolumn;
    const gert::Shape x1_ori_shape = x1_shape->GetStorageShape();
    size_t index_dims = x1_ori_shape.GetDimNum();
    Idcolumn = 1;
    for (size_t i = 0; i < index_dims; i++) {
        Idcolumn *= x1_ori_shape.GetDim(i);
    }
    Tablecolumn = x2_shape->GetStorageShape().GetDim(0);
    Tablerow = x2_shape->GetStorageShape().GetDim(1);
    tiling.set_Idcolumn(Idcolumn);
    tiling.set_Tablerow(Tablerow);
    tiling.set_Tablecolumn(Tablecolumn);

    uint32_t ub_size = 192 * 1024;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ub_size = (uint32_t)((double)ub_size * 0.9);
    const uint32_t ub_reserve_size = 1031 * 4;
    ub_size -= ub_reserve_size;

    uint32_t BLOCK_SIZE = 32;
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = (coreNum < Idcolumn) ? coreNum : Idcolumn;
    if (coreNum == 0) {
        std::cout << "platform aicore num is 0" << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (Idcolumn / coreNum <= 50) {
        coreNum = (Idcolumn / 50 == 0) ? 1 : (Idcolumn / 50);
    }

    uint32_t min_single_core_compute_part = Idcolumn / coreNum;
    uint32_t max_single_core_compute_part = min_single_core_compute_part + 1;
    uint32_t min_part_num = coreNum - (Idcolumn - (Idcolumn / coreNum) * coreNum);

    uint32_t id_column_copyin;
    if (idx_size == 0) {
        std::cout << "idx_size is 0" << std::endl;
        return ge::GRAPH_FAILED;
    }
    const int32_t alignment = 32;
    if (min_part_num != coreNum) {
        id_column_copyin = ((max_single_core_compute_part * (idx_size)) % alignment != 0)
                               ? (((max_single_core_compute_part * idx_size) / alignment + 1) * alignment / idx_size)
                               : max_single_core_compute_part;
    } else {
        id_column_copyin = ((min_single_core_compute_part * (idx_size)) % alignment != 0)
                               ? (((min_single_core_compute_part * idx_size) / alignment + 1) * alignment / idx_size)
                               : min_single_core_compute_part;
    }
    if (table_size == 0) {
        std::cout << "table_size is 0" << std::endl;
        return ge::GRAPH_FAILED;
    }
    uint32_t table_row_copyin = ((Tablerow * (table_size)) % alignment != 0)
                                    ? (((Tablerow * table_size) / alignment + 1) * alignment / table_size)
                                    : Tablerow;
    tiling.set_IdColumnCopyin(id_column_copyin);
    tiling.set_TableRowCopyin(table_row_copyin);

    uint32_t row_in_ub = (ub_size - id_column_copyin * idx_size - Tablerow * table_size) / 2 / (Tablerow * table_size);
    tiling.set_RowInUb(row_in_ub);

    if (row_in_ub == 0) {
        std::cout << "row in ub is 0" << std::endl;
        return ge::GRAPH_FAILED;
    }
    uint32_t min_times_copy_out = (min_single_core_compute_part % row_in_ub == 0)
                                      ? (min_single_core_compute_part / row_in_ub)
                                      : (min_single_core_compute_part / row_in_ub + 1);
    uint32_t max_times_copy_out = (max_single_core_compute_part % row_in_ub == 0)
                                      ? (max_single_core_compute_part / row_in_ub)
                                      : (max_single_core_compute_part / row_in_ub + 1);
    tiling.set_MinSingleCoreComputePart(min_single_core_compute_part);
    tiling.set_MaxSingleCoreComputePart(max_single_core_compute_part);
    tiling.set_MinPartNum(min_part_num);
    tiling.set_MinTimesCopyOut(min_times_copy_out);
    tiling.set_MaxTimesCopyOut(max_times_copy_out);

    if (Tablerow * table_size % alignment == 0 &&
        (Tablecolumn * Tablerow * table_size + id_column_copyin * idx_size + id_column_copyin * Tablerow * table_size <=
         ubSize)) {
        context->SetTilingKey(1);
    }
    if (Tablerow * table_size % alignment == 0 &&
        (Tablecolumn * Tablerow * table_size + id_column_copyin * idx_size + id_column_copyin * Tablerow * table_size >
         ubSize)) {
        context->SetTilingKey(2);
    }
    if (Tablerow * table_size % alignment != 0) {
        context->SetTilingKey(3);
    }
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 1;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    size_t index_dims = x1_shape->GetDimNum();
    const gert::Shape* x2_shape = context->GetInputShape(1);
    auto row = x2_shape->GetDim(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(0);
    for (size_t i = 0; i < index_dims; i++) {
        y_shape->AppendDim(x1_shape->GetDim(i));
    }
    y_shape->AppendDim(row);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(1);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DefaultGather : public OpDef {
public:
    explicit DefaultGather(const char* name) : OpDef(name)
    {
        this->Input("id")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(DefaultGather);
}  // namespace ops
