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
#include <memory>
#include <iostream>
#include <set>

#include "register/register_custom_pass.h"
#include "inc/op_proto.h"
#include "all_ops.h"

using namespace ge;

namespace {
constexpr const char* const kTypeGather = "GatherV2";
constexpr const char* const kTypeWhere = "Where";
constexpr const char* const kTypeGreaterEqual = "GreaterEqual";
constexpr const char* const kTypeReshape = "Reshape";
constexpr const char* const kTypeScatter = "ScatterNd";
constexpr const char* const kTypeSqueeze = "Squeeze";
constexpr const char* const GreaterLeftInputKey = "GreaterLeftInput";
constexpr const char* const GreaterRightInputKey = "GreaterRightInput";
constexpr const char* const kTypeCustomGather = "Default_Gather";
}  // namespace

void GetMatchNodesMap(GNode& great_equal_node, GNode& scatter_node,
                      std::map<std::string, std::shared_ptr<GNode>>& node_type_to_gnode)
{
    std::map<std::string, std::shared_ptr<GNode>> nodes_map;
    auto greater_left_input = great_equal_node.GetInDataNodesAndPortIndexs(0).first;
    auto greater_right_input = great_equal_node.GetInDataNodesAndPortIndexs(1).first;
    nodes_map[GreaterLeftInputKey] = greater_left_input;
    nodes_map[GreaterRightInputKey] = greater_right_input;
    std::shared_ptr<GNode> great_equal_node_ptr = std::make_shared<GNode>(great_equal_node);
    std::shared_ptr<GNode> scatter_node_ptr = std::make_shared<GNode>(scatter_node);
    nodes_map[kTypeGreaterEqual] = great_equal_node_ptr;
    nodes_map[kTypeScatter] = scatter_node_ptr;
    node_type_to_gnode = nodes_map;
}

void FindNodesCanFusion(GraphPtr& graph, std::vector<std::map<std::string, std::shared_ptr<GNode>>>& node_can_fusion)
{
    for (GNode& node : graph->GetAllNodes()) {
        AscendString type;

        node.GetType(type);
        if (type != kTypeScatter) {
            continue;
        }

        auto node1 = node.GetInDataNodesAndPortIndexs(1).first;
        node1->GetType(type);
        if (type != kTypeGather) {
            continue;
        }

        auto node2 = node1->GetInDataNodesAndPortIndexs(1).first;
        node2->GetType(type);
        if (type != kTypeGather) {
            continue;
        }

        auto node3 = node2->GetInDataNodesAndPortIndexs(1).first;
        node3->GetType(type);
        if (type != kTypeSqueeze) {
            continue;
        }

        auto node4 = node3->GetInDataNodesAndPortIndexs(0).first;
        node4->GetType(type);
        if (type != kTypeWhere) {
            continue;
        }

        auto node5 = node4->GetInDataNodesAndPortIndexs(0).first;
        node5->GetType(type);
        if (type != kTypeGreaterEqual) {
            continue;
        }

        std::map<std::string, std::shared_ptr<GNode>> node_type_to_gnodes;
        GetMatchNodesMap(*node5, node, node_type_to_gnodes);
        if (!node_type_to_gnodes.empty()) {
            node_can_fusion.push_back(node_type_to_gnodes);
        }
    }
}

graphStatus ReplaceOpWithFusionOp(GraphPtr& graph, std::map<std::string, std::shared_ptr<GNode>>& node_type_to_gnodes,
                                  CustomPassContext& context, int64_t& replaced_count)
{
    auto scatter_node = node_type_to_gnodes.at(kTypeScatter);
    auto great_equal_node = node_type_to_gnodes.at(kTypeGreaterEqual);
    ge::AscendString name1;
    scatter_node->GetName(name1);

    auto gather_with_sc_node = scatter_node->GetInDataNodesAndPortIndexs(1).first;
    auto gather_up_gather_node = gather_with_sc_node->GetInDataNodesAndPortIndexs(1).first;
    auto table_node = gather_with_sc_node->GetInDataNodesAndPortIndexs(0).first;
    auto squeeze_node = gather_up_gather_node->GetInDataNodesAndPortIndexs(1).first;
    auto where_node = squeeze_node->GetInDataNodesAndPortIndexs(0).first;

    AscendString custom_gather;
    if (gather_up_gather_node->GetName(custom_gather) != GRAPH_SUCCESS) {
        std::cout << "get gather upper gather name failed" << std::endl;
        return GRAPH_FAILED;
    };
    std::string fusion_node_name = std::string(custom_gather.GetString()) + "/" + kTypeCustomGather;
    auto custom_gather_op = op::DefaultGather(fusion_node_name.c_str());
    GNode fusion_node = graph->AddNodeByOp(custom_gather_op);

    auto input_node_a = great_equal_node->GetInDataNodesAndPortIndexs(0).first;
    TensorDesc input_tensor_desc_a;
    TensorDesc table_node_desc;
    input_node_a->GetOutputDesc(0, input_tensor_desc_a);
    table_node->GetOutputDesc(0, table_node_desc);
    if (graph->AddDataEdge(*input_node_a, 0, fusion_node, 0) != GRAPH_SUCCESS) {
        std::cout << "add input 0 to fusion node failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (fusion_node.UpdateInputDesc(0, input_tensor_desc_a) != GRAPH_SUCCESS) {
        std::cout << "fusion node update input 0 desc failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->AddDataEdge(*table_node, 0, fusion_node, 1) != GRAPH_SUCCESS) {
        std::cout << "add table to fusion node failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (fusion_node.UpdateInputDesc(1, table_node_desc) != GRAPH_SUCCESS) {
        std::cout << "fusion node update input 1 desc failed" << std::endl;
        return GRAPH_FAILED;
    };

    auto output_nodes = scatter_node->GetOutDataNodesAndPortIndexs(0);
    ge::AscendString node_name;
    scatter_node->GetName(node_name);
    for (const auto& output_node : output_nodes) {
        output_node.first->GetName(node_name);
        TensorDesc output_tensor_desc;
        scatter_node->GetOutputDesc(0, output_tensor_desc);
        if (graph->RemoveEdge(*scatter_node, 0, *output_node.first, output_node.second) != GRAPH_SUCCESS) {
            std::cout << "remove scatter_node to output edge failed" << std::endl;
            return GRAPH_FAILED;
        };
        if (graph->AddDataEdge(fusion_node, 0, *output_node.first, output_node.second) != GRAPH_SUCCESS) {
            std::cout << "add fusion node to output edge failed" << std::endl;
            return GRAPH_FAILED;
        };
        if (fusion_node.UpdateOutputDesc(0, output_tensor_desc) != GRAPH_SUCCESS) {
            std::cout << "update fusion node output desc failed" << std::endl;
            return GRAPH_FAILED;
        };
    }

    if (scatter_node != nullptr) {
        if (graph->RemoveNode(*scatter_node) != GRAPH_SUCCESS) {
            std::cout << "remove scatter node failed" << std::endl;
            return GRAPH_FAILED;
        };
    }
    if (great_equal_node != nullptr) {
        if (graph->RemoveNode(*great_equal_node) != GRAPH_SUCCESS) {
            std::cout << "remove greater equal node failed" << std::endl;
            return GRAPH_FAILED;
        };
    }
    if (gather_with_sc_node != nullptr) {
        if (graph->RemoveNode(*gather_with_sc_node) != GRAPH_SUCCESS) {
            std::cout << "remove lower gather node failed" << std::endl;
            return GRAPH_FAILED;
        };
    }
    if (gather_up_gather_node != nullptr) {
        if (graph->RemoveNode(*gather_up_gather_node) != GRAPH_SUCCESS) {
            std::cout << "remove upper gather node failed" << std::endl;
            return GRAPH_FAILED;
        };
    }
    if (squeeze_node != nullptr) {
        if (graph->RemoveNode(*squeeze_node) != GRAPH_SUCCESS) {
            std::cout << "remove squeeze node failed" << std::endl;
            return GRAPH_FAILED;
        };
    }
    if (where_node != nullptr) {
        if (graph->RemoveNode(*where_node) != GRAPH_SUCCESS) {
            std::cout << "remove where node failed" << std::endl;
            return GRAPH_FAILED;
        };
    }

    replaced_count += 1;
    return GRAPH_SUCCESS;
}

graphStatus DefaultGatherPass(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------DefaultGatherPass begin----------" << std::endl;
    std::vector<std::map<std::string, std::shared_ptr<GNode>>> node_list;
    FindNodesCanFusion(graph, node_list);

    if (node_list.empty()) {
        std::cout << "Not found DefaultGather node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;
    for (auto& node_type_to_gnodes : node_list) {
        if (ReplaceOpWithFusionOp(graph, node_type_to_gnodes, context, replaced_count) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        };
    }
    std::cout << "DefaultGatherPass end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("DefaultGatherPass").CustomPassFn(DefaultGatherPass);
