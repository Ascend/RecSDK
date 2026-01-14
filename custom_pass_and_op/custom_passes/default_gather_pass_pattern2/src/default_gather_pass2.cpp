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
#include <iostream>
#include <set>

#include "register/register_custom_pass.h"
#include "inc/op_proto.h"
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
constexpr const char* const kTypeTile = "Tile";
constexpr const char* const kTypeSelectV2 = "SelectV2";
constexpr const char* const kTypeExpandDims = "ExpandDims";
constexpr const char* const kTypeGreaterEqual = "GreaterEqual";
constexpr const char* const kTypeGatherV2 = "GatherV2";
constexpr const char* const kTypeMaximum = "Maximum";
constexpr const char* const kTypeData = "Data";
constexpr const char* const kTypeConst = "Const";
constexpr const char* const kTypeConstant = "Constant";
constexpr const char* const kTypeDefaultGather = "DefaultGather";
}  // namespace

void FindNodesCanFusion(GraphPtr& graph, std::vector<GNode>& nodes_can_fusion)
{
    for (const GNode& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeSelectV2) {
            nodes_can_fusion.emplace_back(node);
        }
    }
}

bool CheckConstNodeType(AscendString node_type)
{
    return node_type != kTypeConst && node_type != kTypeConstant;
}

graphStatus ReplaceOpWithFusionOp(GraphPtr& graph, GNode& node, CustomPassContext& context, int64_t& replaced_count)
{
    auto select_input_node0 = node.GetInDataNodesAndPortIndexs(0).first;
    AscendString select_input_type0;
    select_input_node0->GetType(select_input_type0);

    auto select_input_node1 = node.GetInDataNodesAndPortIndexs(1).first;
    AscendString select_input_type1;
    AscendString gather_name;
    select_input_node1->GetType(select_input_type1);
    select_input_node1->GetName(gather_name);

    auto select_input_node2 = node.GetInDataNodesAndPortIndexs(2).first;
    AscendString select_input_type2;
    select_input_node2->GetType(select_input_type2);

    if (select_input_type0 != kTypeExpandDims || select_input_type1 != kTypeGatherV2 ||
        CheckConstNodeType(select_input_type2)) {
        return GRAPH_SUCCESS;
    }

    auto expand_dims_input0 = select_input_node0->GetInDataNodesAndPortIndexs(0).first;
    AscendString expand_dims_input_type0;
    expand_dims_input0->GetType(expand_dims_input_type0);

    if (expand_dims_input_type0 != kTypeGreaterEqual) {
        return GRAPH_SUCCESS;
    }

    auto greater_equal_input0 = expand_dims_input0->GetInDataNodesAndPortIndexs(0).first;
    AscendString greater_equal_input_name0;
    greater_equal_input0->GetName(greater_equal_input_name0);

    auto greater_equal_input1 = expand_dims_input0->GetInDataNodesAndPortIndexs(1).first;
    AscendString greater_equal_input_type1;
    greater_equal_input1->GetType(greater_equal_input_type1);

    if (CheckConstNodeType(greater_equal_input_type1)) {
        return GRAPH_SUCCESS;
    }

    auto gather_table = select_input_node1->GetInDataNodesAndPortIndexs(0).first;

    auto gather_input_1 = select_input_node1->GetInDataNodesAndPortIndexs(1).first;
    AscendString gather_input_type1;
    gather_input_1->GetType(gather_input_type1);

    auto gather_input_2 = select_input_node1->GetInDataNodesAndPortIndexs(2).first;
    AscendString gather_input_type2;
    gather_input_2->GetType(gather_input_type2);

    if (gather_input_type1 != kTypeMaximum) {
        return GRAPH_SUCCESS;
    }

    auto maximum_input_0 = gather_input_1->GetInDataNodesAndPortIndexs(0).first;
    AscendString maximum_input_name0;
    AscendString maximum_input_type0;
    maximum_input_0->GetName(maximum_input_name0);
    maximum_input_0->GetType(maximum_input_type0);

    auto maximum_input_1 = gather_input_1->GetInDataNodesAndPortIndexs(1).first;
    AscendString maximum_input_type1;
    maximum_input_1->GetType(maximum_input_type1);

    if (CheckConstNodeType(maximum_input_type1)) {
        return GRAPH_SUCCESS;
    }

    if (maximum_input_name0 != greater_equal_input_name0) {
        return GRAPH_SUCCESS;
    }

    std::string fusion_node_name = std::string(gather_name.GetString()) + "/" + kTypeDefaultGather;
    auto default_gather_op = op::DefaultGather(fusion_node_name.c_str());
    GNode fusion_node = graph->AddNodeByOp(default_gather_op);
    if (graph->AddDataEdge(*greater_equal_input0, 0, fusion_node, 0) != GRAPH_SUCCESS) {
        std::cout << "add edge greater equal input to fusion node failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->AddDataEdge(*gather_table, 0, fusion_node, 1) != GRAPH_SUCCESS) {
        std::cout << "add edge gather table to fusion node failed" << std::endl;
        return GRAPH_FAILED;
    };

    auto output_nodes = node.GetOutDataNodesAndPortIndexs(0);
    for (const auto& output_node : output_nodes) {
        if (graph->RemoveEdge(node, 0, *output_node.first, output_node.second) != GRAPH_SUCCESS) {
            std::cout << "remove output edge failed" << std::endl;
            return GRAPH_FAILED;
        };
        if (graph->AddDataEdge(fusion_node, 0, *output_node.first, output_node.second) != GRAPH_SUCCESS) {
            std::cout << "add edge fusion node to output edge failed" << std::endl;
            return GRAPH_FAILED;
        };
    }

    if (graph->RemoveNode(node) != GRAPH_SUCCESS) {
        std::cout << "remove select op failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(*select_input_node0) != GRAPH_SUCCESS) {
        std::cout << "remove select input 0 failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(*select_input_node1) != GRAPH_SUCCESS) {
        std::cout << "remove select input 1 failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(*select_input_node2) != GRAPH_SUCCESS) {
        std::cout << "remove select input 2 failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(*expand_dims_input0) != GRAPH_SUCCESS) {
        std::cout << "remove expand_dim failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(*gather_input_1) != GRAPH_SUCCESS) {
        std::cout << "remove gather input 1 node failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(*gather_input_2) != GRAPH_SUCCESS) {
        std::cout << "remove gather input 2 node failed" << std::endl;
        return GRAPH_FAILED;
    };

    replaced_count += 1;

    return GRAPH_SUCCESS;
}

graphStatus DefaultGatherPass2(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------DefaultGatherPass pattern2 begin----------" << std::endl;
    std::vector<GNode> node_list;
    FindNodesCanFusion(graph, node_list);

    if (node_list.empty()) {
        std::cout << "Not found SelectV2 node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;
    for (GNode& node : node_list) {
        if (ReplaceOpWithFusionOp(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        };
    }
    std::cout << "DefaultGatherPass2 end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("DefaultGatherPass2").CustomPassFn(DefaultGatherPass2);
