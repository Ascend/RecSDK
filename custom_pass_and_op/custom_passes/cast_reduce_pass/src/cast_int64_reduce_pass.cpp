/*-*- coding: utf-8 -*-
Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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
#include <iostream>
#include <set>

#include "register/register_custom_pass.h"
#include "all_ops.h"

using namespace ge;

namespace {
constexpr const char* const kTypeReduceSumD = "ReduceSumD";
constexpr const char* const kTypeCast = "Cast";
constexpr const char* const kTypeReduceSum = "ReduceSum";
constexpr const char* const kTypeConst = "Const";
constexpr const char* const kTypeConstant = "Constant";
}  // namespace

void FindNodesCanFusion(GraphPtr& graph, std::vector<GNode>& nodes_can_fusion)
{
    for (const GNode& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeReduceSum) {
            nodes_can_fusion.emplace_back(node);
        }
    }
}

graphStatus ReplaceOp(GraphPtr& graph, GNode& reduce_node, GNode& cast_node)
{
    TensorDesc reduce_input_desc;
    if (reduce_node.GetInputDesc(0, reduce_input_desc) != GRAPH_SUCCESS) {
        std::cout << "get reduce input desc0 failed" << std::endl;
        return GRAPH_FAILED;
    }
    reduce_input_desc.SetDataType(DT_INT32);
    if (reduce_node.UpdateInputDesc(0, reduce_input_desc) != GRAPH_SUCCESS) {
        std::cout << "update reduce input desc0 failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc reduce_output_desc;
    if (reduce_node.GetOutputDesc(0, reduce_output_desc) != GRAPH_SUCCESS) {
        std::cout << "get reduce output desc0 failed" << std::endl;
        return GRAPH_FAILED;
    }
    reduce_output_desc.SetDataType(DT_INT32);
    if (reduce_node.UpdateOutputDesc(0, reduce_output_desc) != GRAPH_SUCCESS) {
        std::cout << "update reduce output desc0 failed" << std::endl;
        return GRAPH_FAILED;
    }
    TensorDesc cast_output_desc;
    if (cast_node.GetOutputDesc(0, cast_output_desc) != GRAPH_SUCCESS) {
        std::cout << "get cast output desc0 failed" << std::endl;
        return GRAPH_FAILED;
    }
    cast_output_desc.SetDataType(DT_INT32);
    if (cast_node.UpdateOutputDesc(0, cast_output_desc) != GRAPH_SUCCESS) {
        std::cout << "update cast output desc0 failed" << std::endl;
        return GRAPH_FAILED;
    }

    int32_t cast_dst_type = 3;  // int32
    cast_node.SetAttr(AscendString("dst_type"), cast_dst_type);

    auto reduce_output_nodes = reduce_node.GetOutDataNodesAndPortIndexs(0);

    AscendString cast_name;
    auto ret = cast_node.GetName(cast_name);
    if (ret != GRAPH_SUCCESS) {
        std::cout << "get cast node name failed" << std::endl;
        return GRAPH_FAILED;
    }

    std::string tail_cast_name = std::string(cast_name.GetString()) + "_2";
    op::Cast tail_cast_op(tail_cast_name);
    GNode tail_cast_gnode = graph->AddNodeByOp(tail_cast_op);

    int32_t tail_cast_dst_type = 9;  // int64
    TensorDesc tail_cast_desc(reduce_output_desc);
    if (tail_cast_gnode.UpdateInputDesc(0, tail_cast_desc) != GRAPH_SUCCESS) {
        std::cout << "update tail cast desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (tail_cast_gnode.SetAttr(AscendString("dst_type"), tail_cast_dst_type) != GRAPH_SUCCESS) {
        std::cout << "set tail cast dest type failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (tail_cast_op.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer tail_op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }

    for (auto out_node : reduce_output_nodes) {
        if (graph->RemoveEdge(reduce_node, 0, *out_node.first, out_node.second) != GRAPH_SUCCESS) {
            std::cout << "remove reduce output edge failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(reduce_node, 0, tail_cast_gnode, 0) != GRAPH_SUCCESS) {
            std::cout << "add edge reduce output to tail cast failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(tail_cast_gnode, 0, *out_node.first, out_node.second) != GRAPH_SUCCESS) {
            std::cout << "add edge tail cast to output failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

graphStatus CheckReduceDims(GNode& reduce_node)
{
    Tensor const_tensor;
    if (reduce_node.GetInputConstData(1, const_tensor) != GRAPH_SUCCESS) {
        std::cout << "get const tensor failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (const_tensor.GetDataType() != DT_INT32) {
        std::cout << "const dtype is not int32" << std::endl;
        return GRAPH_FAILED;
    }

    size_t reduce_dims_size = const_tensor.GetSize();
    uint8_t* reduce_dims_data = const_tensor.GetData();

    if (reduce_dims_size == 0 || reduce_dims_data == nullptr || reduce_dims_size % 4 != 0) {
        std::cout << "check reduce dims failed" << std::endl;
        return GRAPH_FAILED;
    }

    reduce_dims_size = reduce_dims_size / 4;
    int32_t* int32_data = reinterpret_cast<int32_t*>(reduce_dims_data);

    int64_t max_val = 1;
    TensorDesc input_desc;
    if (reduce_node.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        std::cout << "get cast input desc failed" << std::endl;
        return GRAPH_SUCCESS;
    }

    Shape input_shape = input_desc.GetShape();
    size_t dims = input_shape.GetDimNum();

    for (size_t i = 0; i < reduce_dims_size; i++) {
        int64_t dim;
        if (int32_data[i] >= static_cast<int32_t>(dims)) {
            std::cout << "unsupported reduce dim" << std::endl;
            return GRAPH_FAILED;
        } else if (int32_data[i] < -1 * static_cast<int32_t>(dims)) {
            std::cout << "unsupported reduce dim" << std::endl;
            return GRAPH_FAILED;
        } else if (int32_data[i] >= 0) {
            dim = input_shape.GetDim(int32_data[i]);
        } else {
            dim = input_shape.GetDim(dims + int32_data[i]);
        }
        max_val *= dim;
    }

    if (max_val > INT32_MAX) {
        std::cout << "max_val excceds int32: " << max_val << std::endl;
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

graphStatus CheckCastDtype(GNode& cast_node)
{
    TensorDesc input_desc;
    if (cast_node.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        std::cout << "get cast input desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    DataType input_dtype = input_desc.GetDataType();
    if (input_dtype != DT_BOOL) {
        return GRAPH_FAILED;
    }

    TensorDesc output_desc;
    if (cast_node.GetOutputDesc(0, output_desc) != GRAPH_SUCCESS) {
        std::cout << "get cast output desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    DataType output_dtype = output_desc.GetDataType();
    if (output_dtype != DT_INT64) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

bool CheckCastOutputNum(GNode& cast_node)
{
    auto cast_output_nodes = cast_node.GetOutDataNodesAndPortIndexs(0);
    return cast_output_nodes.size() == 1;
}

graphStatus ReplaceOpWithFusionOp(GraphPtr& graph, GNode& node, CustomPassContext& context, int64_t& replaced_count)
{
    auto reduce_input_node0 = node.GetInDataNodesAndPortIndexs(0).first;
    AscendString reduce_input_type0;
    reduce_input_node0->GetType(reduce_input_type0);

    auto reduce_input_node1 = node.GetInDataNodesAndPortIndexs(1).first;
    AscendString reduce_input_type1;
    reduce_input_node1->GetType(reduce_input_type1);

    if (reduce_input_type0 != kTypeCast) {
        return GRAPH_SUCCESS;
    }

    if (reduce_input_type1 == kTypeConst && reduce_input_type0 == kTypeConstant) {
        return GRAPH_SUCCESS;
    }

    if (CheckCastDtype(*reduce_input_node0) != GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    }

    if (!CheckCastOutputNum(*reduce_input_node0)) {
        return GRAPH_SUCCESS;
    }

    if (CheckReduceDims(node) != GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    }

    if (ReplaceOp(graph, node, *reduce_input_node0) != GRAPH_SUCCESS) {
        std::cout << "replace reduce node failed" << std::endl;
        return GRAPH_FAILED;
    }
    replaced_count += 1;

    return GRAPH_SUCCESS;
}

graphStatus CastInt64ReducePass(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------Cast Int64 Reduce Pass begin----------" << std::endl;
    std::vector<GNode> node_list;
    FindNodesCanFusion(graph, node_list);

    if (node_list.empty()) {
        std::cout << "Not found Cast Int64 Reduce Pass node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;
    for (GNode& node : node_list) {
        if (ReplaceOpWithFusionOp(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            std::cout << "replace fusion op failed" << std::endl;
        }
    }
    std::cout << "Cast Int64 Reduce Custom end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("CastInt64ReducePass").CustomPassFn(CastInt64ReducePass).Stage(CustomPassStage::kAfterInferShape);
