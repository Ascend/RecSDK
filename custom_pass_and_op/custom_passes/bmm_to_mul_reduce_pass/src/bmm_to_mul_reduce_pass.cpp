/* -*- coding: utf-8 -*-
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
constexpr const char* const kTypeSub = "Sub";
constexpr const char* const kTypeAdd = "Add";
constexpr const char* const kTypeSqrt = "Sqrt";
constexpr const char* const kTypeRealDiv = "RealDiv";
constexpr const char* const kTypeBatchNorm = "BatchNorm";
constexpr const char* const kTypeReshape = "Reshape";
constexpr const char* const kTypeBatchMatmul = "BatchMatMulV2";
constexpr const char* const kTypePack = "Pack";
constexpr const char* const kTypeTranspose = "Transpose";
constexpr const char* const kTypeSoftmax = "SoftmaxV2";
constexpr const char* const kTypeMul = "Mul";
constexpr const char* const kTypeReduceSumD = "ReduceSumD";
}  // namespace

void FindNodesCanFusion(GraphPtr& graph, std::vector<GNode>& nodes_can_fusion)
{
    for (const GNode& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeBatchMatmul) {
            nodes_can_fusion.emplace_back(node);
        }
    }
}

graphStatus ReplaceNode(GraphPtr& graph, const char* name_prefix, std::vector<GNodePtr>& delete_nodes,
                        std::pair<GNodePtr, int32_t> left_input, TensorDesc& left_desc,
                        std::pair<GNodePtr, int32_t> right_input, TensorDesc& right_desc,
                        std::vector<std::pair<GNodePtr, int32_t>>& outputs)
{
    std::string mul_node_name = std::string(name_prefix) + "/" + kTypeMul;
    auto mul = op::Mul(mul_node_name.c_str());
    GNode mul_node = graph->AddNodeByOp(mul);
    if (mul.UpdateInputDesc(static_cast<uint32_t>(0), left_desc) != GRAPH_SUCCESS) {
        std::cout << "update mul left op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (mul.UpdateInputDesc(static_cast<uint32_t>(1), right_desc) != GRAPH_SUCCESS) {
        std::cout << "update mul right op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (mul.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infershape mul node failed" << std::endl;
        return GRAPH_FAILED;
    }
    TensorDesc mul_output_desc = mul.GetOutputDesc(static_cast<uint32_t>(0));

    std::string reduce_node_name = std::string(name_prefix) + "/" + kTypeReduceSumD;
    auto reduce = op::ReduceSum(reduce_node_name.c_str());
    AscendString keep_dims = "keep_dims";
    GNode reduce_node = graph->AddNodeByOp(reduce);
    bool keep_dims_attr = true;
    reduce_node.SetAttr(keep_dims, keep_dims_attr);

    int64_t reduce_axis = -1;
    TensorDesc reduce_axis_tensordesc(ge::Shape({1}), FORMAT_ND, DT_INT64);
    reduce_axis_tensordesc.SetOriginShape(ge::Shape({1}));
    Tensor reduce_axis_tensor(reduce_axis_tensordesc);
    reduce_axis_tensor.SetData(reinterpret_cast<uint8_t*>(&reduce_axis), sizeof(int64_t));
    auto axis = op::Const("axis").set_attr_value(reduce_axis_tensor);
    GNode axis_node = graph->AddNodeByOp(axis);
    TensorDesc axis_op_desc(reduce_axis_tensordesc);
    if (axis.UpdateOutputDesc(static_cast<uint32_t>(0), axis_op_desc) != GRAPH_SUCCESS) {
        std::cout << "update axis op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (axis.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infershape to reduce axis node failed" << std::endl;
        return GRAPH_FAILED;
    }
    reduce.SetInput("axes", axis);

    TensorDesc reduce_axis_desc(reduce_axis_tensordesc);
    TensorDesc reduce_data_desc(mul_output_desc);
    if (reduce.UpdateInputDesc(static_cast<uint32_t>(0), reduce_data_desc) != GRAPH_SUCCESS) {
        std::cout << "update reduce data op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (reduce.UpdateInputDesc(static_cast<uint32_t>(1), reduce_axis_desc) != GRAPH_SUCCESS) {
        std::cout << "update reduce axis op desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (reduce.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer shape for reduce node failed" << std::endl;
        return GRAPH_FAILED;
    }
    reduce.BreakConnect();

    if (graph->AddDataEdge(mul_node, 0, reduce_node, 0) != GRAPH_SUCCESS) {
        std::cout << "add edge mul to reduce node failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(axis_node, 0, reduce_node, 1) != GRAPH_SUCCESS) {
        std::cout << "add edge axis to reduce node failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(*left_input.first, left_input.second, mul_node, 0) != GRAPH_SUCCESS) {
        std::cout << "add edge left input to mul failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(*right_input.first, right_input.second, mul_node, 1) != GRAPH_SUCCESS) {
        std::cout << "add edge right input to mul failed" << std::endl;
        return GRAPH_FAILED;
    }

    for (auto node : delete_nodes) {
        if (graph->RemoveNode(*node) != GRAPH_SUCCESS) {
            std::cout << "delete node failed" << std::endl;
            return GRAPH_FAILED;
        }
    }
    for (auto output : outputs) {
        if (graph->AddDataEdge(reduce_node, 0, *output.first, output.second) != GRAPH_SUCCESS) {
            std::cout << "add edge reduce to output failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

graphStatus ReplaceOpWithFusionOp(GraphPtr& graph, GNode& node, CustomPassContext& context, int64_t& replaced_count)
{
    auto bmm_input0_node = node.GetInDataNodesAndPortIndexs(0);
    auto bmm_input1_node = node.GetInDataNodesAndPortIndexs(1);
    if (bmm_input0_node.first == nullptr) {
        std::cout << "bmm does not have input 0" << std::endl;
        return GRAPH_FAILED;
    }
    if (bmm_input1_node.first == nullptr) {
        std::cout << "bmm does not have input 1" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc bmm_input0_desc;
    TensorDesc bmm_input1_desc;
    if (node.GetInputDesc(0, bmm_input0_desc) != GRAPH_SUCCESS) {
        std::cout << "get bmm input0 desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (node.GetInputDesc(1, bmm_input1_desc) != GRAPH_SUCCESS) {
        std::cout << "get bmm input1 desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    Shape bmm_input0_shape = bmm_input0_desc.GetShape();
    Shape bmm_input1_shape = bmm_input1_desc.GetShape();

    uint32_t k_index;
    uint32_t n_index;
    bool transpose_a;
    bool transpose_b;
    uint32_t dims = bmm_input0_shape.GetDimNum();
    if (node.GetAttr(AscendString("adj_x1"), transpose_a) != GRAPH_SUCCESS) {
        std::cout << "bmm node get adj_x1 attr failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (node.GetAttr(AscendString("adj_x2"), transpose_b) != GRAPH_SUCCESS) {
        std::cout << "bmm node get adj_x2 attr failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (transpose_b) {
        n_index = dims - 2;
        k_index = dims - 1;
    } else {
        n_index = dims - 1;
        k_index = dims - 2;
    }

    if (bmm_input1_shape.GetDim(n_index) != 1 || bmm_input1_shape.GetDim(k_index) > 16) {
        return GRAPH_SUCCESS;
    }

    std::vector<GNodePtr> delete_nodes;
    delete_nodes.emplace_back(std::make_shared<GNode>(node));

    auto output_nodes = node.GetOutDataNodesAndPortIndexs(0);
    if (output_nodes.size() != 1U) {
        std::cout << "Outputs size not equal to 1" << std::endl;
        return GRAPH_SUCCESS;
    }

    if (transpose_a) {
        std::cout << "left matrix has been transposed" << std::endl;
        return GRAPH_SUCCESS;
    }

    AscendString bmm_name;
    node.GetName(bmm_name);

    if (transpose_b) {
        replaced_count += 1;
        return ReplaceNode(graph, bmm_name.GetString(), delete_nodes, bmm_input0_node, bmm_input0_desc, bmm_input1_node,
                           bmm_input1_desc, output_nodes);
    }

    AscendString bmm_input1_type;
    bmm_input1_node.first->GetType(bmm_input1_type);

    if (bmm_input1_type != kTypeTranspose) {
        return GRAPH_SUCCESS;
    }

    auto transpose_input0_node = bmm_input1_node.first->GetInDataNodesAndPortIndexs(0);
    TensorDesc transpose_input0_desc;
    if (transpose_input0_node.first->GetInputDesc(0, transpose_input0_desc) != GRAPH_SUCCESS) {
        std::cout << "get transpose input desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    Shape transpose_input0_shape = transpose_input0_desc.GetShape();

    for (uint32_t i = 0; i < dims - 2; i++) {
        if (bmm_input1_shape.GetDim(i) != transpose_input0_shape.GetDim(i) && bmm_input1_shape.GetDim(i) != 1 &&
            transpose_input0_shape.GetDim(i) != 1) {
            std::cout << "batch dim is not the same, i is: " << i
                      << "    bmm_input dim is: " << bmm_input1_shape.GetDim(i)
                      << "    transpose_input_dims is: " << transpose_input0_shape.GetDim(i) << std::endl;
            return GRAPH_SUCCESS;
        }
    }

    if (bmm_input1_shape.GetDim(dims - 1) != transpose_input0_shape.GetDim(dims - 2) ||
        bmm_input1_shape.GetDim(dims - 2) != transpose_input0_shape.GetDim(dims - 1)) {
        std::cout << "last two dims are not the same, bmm_input dim is: "
                  << bmm_input1_shape.GetDim(dims - 2) << ", " << bmm_input1_shape.GetDim(dims - 1)
                  << "    transpose_input_dims is: " << transpose_input0_shape.GetDim(dims - 2) << ", "
                  << transpose_input0_shape.GetDim(dims - 1) << std::endl;
        return GRAPH_SUCCESS;
    }

    delete_nodes.emplace_back(bmm_input1_node.first);

    replaced_count += 1;
    return ReplaceNode(graph, bmm_name.GetString(), delete_nodes, bmm_input0_node, bmm_input0_desc,
                       transpose_input0_node, transpose_input0_desc, output_nodes);
}

graphStatus BatchMatmulToMulReducePass(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------BatchMatmulToMulReducePass begin----------" << std::endl;
    std::vector<GNode> node_list;
    FindNodesCanFusion(graph, node_list);

    if (node_list.empty()) {
        std::cout << "Not found BatchMatmul node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;
    for (GNode& node : node_list) {
        if (ReplaceOpWithFusionOp(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        };
    }
    std::cout << "BatchMatmulToMulReducePass end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("BatchMatmulToMulReducePass")
    .CustomPassFn(BatchMatmulToMulReducePass)
    .Stage(CustomPassStage::kAfterInferShape);
;
