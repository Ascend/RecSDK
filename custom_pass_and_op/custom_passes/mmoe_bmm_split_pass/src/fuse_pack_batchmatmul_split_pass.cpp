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
#include "register_custom_pass.h"
#include "all_ops.h"

using namespace std;
using namespace ge;

namespace {

constexpr const char* kOpNameData = "Data";
constexpr const char* kOpNameConst = "Const";
constexpr const char* kOpNameConstant = "Constant";
constexpr const char* kOpNameMatmulV2 = "MatMulV2";
constexpr const char* kOpNameMatmulV3 = "MatMulV3";
constexpr const int32_t RecursionSearchMaxLevel = 10;

GNode FindLastNode(GNode& node, int32_t index)
{
    std::pair<GNodePtr, int32_t> inputNode = node.GetInDataNodesAndPortIndexs(index);
    ge::GNode& input_node = *inputNode.first;
    return input_node;
}

bool CheckNodeType(GNode& node, const char* node_type)
{
    ge::AscendString type;
    node.GetType(type);
    std::string out_node_type(type.GetString());
    std::string std_string(node_type);
    if (out_node_type == node_type) {
        return true;
    }
    return false;
}

}  // namespace

bool RecursionSearchConcatNode(GNode& src_node, GNode& root_node, int32_t search_level, int32_t& matmul_node_level)
{
    // 递归结束条件
    if (search_level > RecursionSearchMaxLevel) {
        return false;
    } else {
        GNode next_node;
        int32_t no_const_node_num = 0;

        if (CheckNodeType(src_node, kOpNameData)) {
            root_node = src_node;
            return true;
        }
        auto node_inputs_size = src_node.GetInputsSize();
        if (node_inputs_size == -1 || node_inputs_size == 0) {
            return false;
        }
        if (CheckNodeType(src_node, kOpNameConst) || CheckNodeType(src_node, kOpNameConstant)) {
            return false;
        }
        for (size_t i = 0; i < node_inputs_size; i++) {
            GNode tmp_node = FindLastNode(src_node, i);
            if (CheckNodeType(tmp_node, kOpNameConst) || CheckNodeType(tmp_node, kOpNameConstant)) {
                continue;
            } else if (no_const_node_num == 0) {
                next_node = tmp_node;
            }
            no_const_node_num++;
        }

        if (no_const_node_num >= 2) {
            root_node = src_node;
            return true;
        }

        if (CheckNodeType(src_node, kOpNameMatmulV2) || CheckNodeType(src_node, kOpNameMatmulV3)) {
            matmul_node_level += 1;
        }

        return RecursionSearchConcatNode(next_node, root_node, search_level + 1, matmul_node_level);
    }
}

void FindSameLevelMatmulNode(GraphPtr& graph, CustomPassContext& custom_context,
                             std::map<std::string, std::map<int32_t, std::vector<GNode>>>& matmul_node_map)
{
    std::vector<GNode> nodes = graph->GetAllNodes();
    for (GNode& node : nodes) {
        if (CheckNodeType(node, kOpNameMatmulV2) || CheckNodeType(node, kOpNameMatmulV3)) {
            int32_t matmul_node_level = 0;
            GNode root_node;
            bool search_concat_flag = RecursionSearchConcatNode(node, root_node, 0, matmul_node_level);
            if (search_concat_flag) {
                ge::AscendString name;
                root_node.GetName(name);
                std::string root_node_name(name.GetString());
                matmul_node_map[root_node_name][matmul_node_level].push_back(node);
            }
        }
    }
}

graphStatus CheckMatmulShape(std::vector<GNode> matmul_node_list, bool& transpose_attr_x1_compare,
                             bool& transpose_attr_x2_compare)
{
    if (matmul_node_list.size() == 0) {
        std::cout << "mamtul_node_list is empty" << std::endl;
        return GRAPH_FAILED;
    }
    GNode node_matmul = matmul_node_list[0];

    if (node_matmul.GetAttr("transpose_x1", transpose_attr_x1_compare) != GRAPH_SUCCESS) {
        std::cout << "get matmul transpose_x1 compare attr failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (node_matmul.GetAttr("transpose_x2", transpose_attr_x2_compare) != GRAPH_SUCCESS) {
        std::cout << "get matmul transpose_x2 compare attr failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc matmul_input_0;

    if (node_matmul.GetInputDesc(0, matmul_input_0) != GRAPH_SUCCESS) {
        std::cout << "get matmul input 0 desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    TensorDesc matmul_input_1;
    if (node_matmul.GetInputDesc(1, matmul_input_1) != GRAPH_SUCCESS) {
        std::cout << "get matmul input 1 desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    Shape matmul_input_0_shape = matmul_input_0.GetShape();
    std::vector<int64_t> matmul_input_0_dims = matmul_input_0_shape.GetDims();
    Shape matmul_input_1_shape = matmul_input_1.GetShape();
    std::vector<int64_t> matmul_input_1_dims = matmul_input_1_shape.GetDims();
    for (int32_t matmul_index = 1; matmul_index < matmul_node_list.size(); matmul_index++) {
        GNode node_matmul = matmul_node_list[matmul_index];
        bool transpose_attr_x1 = false;
        bool transpose_attr_x2 = false;
        if (node_matmul.GetAttr("transpose_x1", transpose_attr_x1) != GRAPH_SUCCESS) {
            std::cout << "get matmul transpose_x1 attr failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (node_matmul.GetAttr("transpose_x2", transpose_attr_x2) != GRAPH_SUCCESS) {
            std::cout << "get matmul transpose_x2 attr failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (transpose_attr_x1_compare != transpose_attr_x1 || transpose_attr_x2_compare != transpose_attr_x2) {
            std::cout << "transpose_x1 or transpose_x2 attr not align" << std::endl;
            return GRAPH_FAILED;
        }

        TensorDesc input_0;
        if (node_matmul.GetInputDesc(0, input_0) != GRAPH_SUCCESS) {
            std::cout << "get matmul input 0 desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        TensorDesc input_1;
        if (node_matmul.GetInputDesc(1, input_1) != GRAPH_SUCCESS) {
            std::cout << "get matmul input 1 desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        Shape input_0_shape = input_0.GetShape();
        std::vector<int64_t> input_0_dims = input_0_shape.GetDims();
        Shape input_1_shape = input_1.GetShape();
        std::vector<int64_t> input_1_dims = input_1_shape.GetDims();
        if (matmul_input_0_dims.size() != input_0_dims.size()) {
            std::cout << "matmul_input_0_dims not equal" << std::endl;
            return GRAPH_FAILED;
        }
        if (input_0_dims.size() < 2) {
            std::cout << "invalid matmul input dims" << std::endl;
            return GRAPH_FAILED;
        }
        if (matmul_input_0_dims[0] != input_0_dims[0] || matmul_input_0_dims[1] != input_0_dims[1]) {
            std::cout << "left input shape does not the same" << std::endl;
            return GRAPH_FAILED;
        }
        if (matmul_input_1_dims[0] != input_1_dims[0] || matmul_input_1_dims[1] != input_1_dims[1]) {
            std::cout << "right input shape does not the same" << std::endl;
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

graphStatus ProcessPackNode(GraphPtr& graph, std::vector<GNode> matmul_node_list, GNode& pack_0_node,
                            GNode& pack_1_node, std::string concat_node_name, int32_t index)
{
    auto pack_0 = op::Pack(concat_node_name + "pack_0_" + std::to_string(index))
                      .create_dynamic_input_x(matmul_node_list.size())
                      .set_attr_N(matmul_node_list.size());
    pack_0_node = graph->AddNodeByOp(pack_0);
    auto pack_1 = op::Pack(concat_node_name + "pack_1_" + std::to_string(index))
                      .create_dynamic_input_x(matmul_node_list.size())
                      .set_attr_N(matmul_node_list.size());
    pack_1_node = graph->AddNodeByOp(pack_1);

    for (int32_t matmul_index = 0; matmul_index < matmul_node_list.size(); matmul_index++) {
        GNode matmul_node = matmul_node_list[matmul_index];

        TensorDesc matmul_input_desc_0;
        TensorDesc matmul_input_desc_1;
        if (matmul_node.GetInputDesc(0, matmul_input_desc_0) != GRAPH_SUCCESS) {
            std::cout << "get matmul input 0 desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (matmul_node.GetInputDesc(1, matmul_input_desc_1) != GRAPH_SUCCESS) {
            std::cout << "get matmul input 1 desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        TensorDesc pack_input_0_desc(matmul_input_desc_0);
        TensorDesc pack_input_1_desc(matmul_input_desc_1);
        pack_0_node.UpdateInputDesc(matmul_index, pack_input_0_desc);
        pack_1_node.UpdateInputDesc(matmul_index, pack_input_1_desc);

        std::pair<GNodePtr, int32_t> matmulInputMap = matmul_node.GetInDataNodesAndPortIndexs(0);
        if (graph->RemoveEdge(*matmulInputMap.first, matmulInputMap.second, matmul_node, 0) != GRAPH_SUCCESS) {
            std::cout << "RemoveEdge matmul 0 and matmul 0 input failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(*matmulInputMap.first, matmulInputMap.second, pack_0_node, matmul_index) !=
            GRAPH_SUCCESS) {
            std::cout << "AddDataEdge pack and matmul 0 input failed" << std::endl;
            return GRAPH_FAILED;
        }

        matmulInputMap = matmul_node.GetInDataNodesAndPortIndexs(1);
        if (matmulInputMap.first == nullptr) {
            std::cout << "get matmul data nodes input gnode failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->RemoveEdge(*matmulInputMap.first, matmulInputMap.second, matmul_node, 1) != GRAPH_SUCCESS) {
            std::cout << "RemoveEdge matmul 1 and matmul 1 input failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->AddDataEdge(*matmulInputMap.first, matmulInputMap.second, pack_1_node, matmul_index) !=
            GRAPH_SUCCESS) {
            std::cout << "AddDataEdge pack and matmul 1 input failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    if (pack_0.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer pack_0 op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (pack_1.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer pack_1 op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus ProcessBatchMatmulNode(GraphPtr& graph, GNode& pack_0_node, GNode& pack_1_node, GNode& batch_matmul_node,
                                   std::string concat_node_name, int32_t index, bool& transpose_attr_x1_compare,
                                   bool& transpose_attr_x2_compare)
{
    auto batch_matmul = op::BatchMatMulV2(concat_node_name + "batch_matmul_" + std::to_string(index));
    batch_matmul_node = graph->AddNodeByOp(batch_matmul);
    batch_matmul_node.SetAttr("adj_x1", transpose_attr_x1_compare);
    batch_matmul_node.SetAttr("adj_x2", transpose_attr_x2_compare);

    TensorDesc pack_ouput_desc_0;
    if (pack_0_node.GetOutputDesc(0, pack_ouput_desc_0) != GRAPH_SUCCESS) {
        std::cout << "get pack_0 output 0 desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc batch_matmul_input_0_desc(pack_ouput_desc_0);
    batch_matmul_node.UpdateInputDesc(0, batch_matmul_input_0_desc);
    TensorDesc pack_ouput_desc_1;
    if (pack_1_node.GetOutputDesc(0, pack_ouput_desc_1) != GRAPH_SUCCESS) {
        std::cout << "get pack_1 output 0 desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc batch_matmul_input_1_desc(pack_ouput_desc_1);
    batch_matmul_node.UpdateInputDesc(1, batch_matmul_input_1_desc);
    if (batch_matmul.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer batch_matmul op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(pack_0_node, 0, batch_matmul_node, 0) != GRAPH_SUCCESS) {
        std::cout << "AddDataEdge pack_0 and batch_matmul failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(pack_1_node, 0, batch_matmul_node, 1) != GRAPH_SUCCESS) {
        std::cout << "AddDataEdge pack_1 and batch_matmul failed" << std::endl;
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus ProcessSplitAndSqueezeNode(GraphPtr& graph, std::vector<GNode> matmul_node_list, GNode& batch_matmul_node,
                                       std::string concat_node_name, int32_t index)
{
    auto split = op::Split(concat_node_name + "split_" + std::to_string(index))
                     .set_attr_num_split(matmul_node_list.size())
                     .create_dynamic_output_y(matmul_node_list.size());
    GNode split_node = graph->AddNodeByOp(split);

    std::string split_const_name = "split_const";
    op::Const split_dim_op(split_const_name.c_str());
    TensorDesc split_dim_desc(ge::Shape({1}), FORMAT_ND, DT_INT32);
    split_dim_desc.SetOriginShape(ge::Shape({1}));
    int32_t split_dims = 0;
    Tensor split_dim_tensor(split_dim_desc, reinterpret_cast<const uint8_t*>(&split_dims), sizeof(int32_t));
    split_dim_op.set_attr_value(split_dim_tensor);
    GNode split_dim_gnode = graph->AddNodeByOp(split_dim_op);
    split.SetInput("split_dim", split_dim_op);
    TensorDesc split_dim_gnode_desc(split_dim_desc);
    if (split_dim_gnode.UpdateOutputDesc(0, split_dim_gnode_desc) != GRAPH_SUCCESS) {
        std::cout << "update split output desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    split_node.UpdateInputDesc(0, split_dim_gnode_desc);
    TensorDesc batch_matmul_ouput_desc;
    if (batch_matmul_node.GetOutputDesc(0, batch_matmul_ouput_desc) != GRAPH_SUCCESS) {
        std::cout << "get batch_matmul output desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    TensorDesc split_input_1_desc(batch_matmul_ouput_desc);
    split_node.UpdateInputDesc(1, split_input_1_desc);
    if (split.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "infer split op shape and type failed" << std::endl;
        return GRAPH_FAILED;
    }
    split.BreakConnect();

    if (graph->AddDataEdge(split_dim_gnode, 0, split_node, 0) != GRAPH_SUCCESS) {
        std::cout << "AddDataEdge split_const and split failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(batch_matmul_node, 0, split_node, 1) != GRAPH_SUCCESS) {
        std::cout << "AddDataEdge batch_matmul and split failed" << std::endl;
        return GRAPH_FAILED;
    }

    for (int32_t matmul_index = 0; matmul_index < matmul_node_list.size(); matmul_index++) {
        GNode matmul_node = matmul_node_list[matmul_index];
        auto squeeze =
            op::Squeeze(concat_node_name + "squeeze_" + std::to_string(index) + "_" + std::to_string(matmul_index));
        GNode squeeze_node = graph->AddNodeByOp(squeeze);

        TensorDesc split_ouput_desc;
        if (split_node.GetOutputDesc(matmul_index, split_ouput_desc) != GRAPH_SUCCESS) {
            std::cout << "get split output desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        TensorDesc squeeze_input_desc(split_ouput_desc);
        squeeze_node.UpdateInputDesc(0, squeeze_input_desc);
        if (squeeze.InferShapeAndType() != GRAPH_SUCCESS) {
            std::cout << "infer squeeze op shape and type failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(split_node, matmul_index, squeeze_node, 0) != GRAPH_SUCCESS) {
            std::cout << "AddDataEdge splt and squeeze failed" << std::endl;
            return GRAPH_FAILED;
        }

        for (int32_t i = 0; i < matmul_node.GetOutputsSize(); i++) {
            std::vector<std::pair<GNodePtr, int32_t>> outputNodes = matmul_node.GetOutDataNodesAndPortIndexs(i);
            for (int32_t j = 0; j < outputNodes.size(); j++) {
                if (graph->RemoveEdge(matmul_node, i, *outputNodes[j].first, outputNodes[j].second) != GRAPH_SUCCESS) {
                    std::cout << "RemoveEdge matmul and matmul output failed" << std::endl;
                    return GRAPH_FAILED;
                }

                if (graph->AddDataEdge(squeeze_node, i, *outputNodes[j].first, outputNodes[j].second) !=
                    GRAPH_SUCCESS) {
                    std::cout << "AddDataEdge squeeze and matmul output failed" << std::endl;
                    return GRAPH_FAILED;
                }
            }
        }

        if (graph->RemoveNode(matmul_node) != GRAPH_SUCCESS) {
            std::cout << "RemoveEdge matmul node failed" << std::endl;
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

graphStatus ReplaceMatmul(GraphPtr& graph, CustomPassContext& custom_context,
                          std::map<std::string, std::map<int32_t, std::vector<GNode>>>& matmul_node_map,
                          int32_t& affect_count)
{
    for (const auto& concat_pair : matmul_node_map) {
        int32_t index = 0;
        std::string concat_node_name = concat_pair.first;
        for (const auto& matmul_list_pair : concat_pair.second) {
            int32_t matmul_level = matmul_list_pair.first;
            std::vector<GNode> matmul_node_list = matmul_list_pair.second;
            if ((matmul_node_list.size() > 1) && (matmul_level != 1)) {
                graphStatus ret = GRAPH_FAILED;
                bool transpose_attr_x1_compare = false;
                bool transpose_attr_x2_compare = false;
                ret = CheckMatmulShape(matmul_node_list, transpose_attr_x1_compare, transpose_attr_x2_compare);
                if (ret != GRAPH_SUCCESS) {
                    continue;
                }

                GNode pack_0_node;
                GNode pack_1_node;
                GNode batch_matmul_node;
                ret = ProcessPackNode(graph, matmul_node_list, pack_0_node, pack_1_node, concat_node_name, index);
                if (ret != GRAPH_SUCCESS) {
                    return GRAPH_FAILED;
                }

                ret = ProcessBatchMatmulNode(graph, pack_0_node, pack_1_node, batch_matmul_node, concat_node_name,
                                             index, transpose_attr_x1_compare, transpose_attr_x2_compare);
                if (ret != GRAPH_SUCCESS) {
                    return GRAPH_FAILED;
                }

                ret = ProcessSplitAndSqueezeNode(graph, matmul_node_list, batch_matmul_node, concat_node_name, index);
                if (ret != GRAPH_SUCCESS) {
                    return GRAPH_FAILED;
                }

                index++;
                affect_count++;
            }
        }
    }
    return GRAPH_SUCCESS;
}

graphStatus FusePackBatchMatmulSplitPass(GraphPtr& graph, CustomPassContext& custom_context)
{
    std::cout << "----------FusePackBatchMatmulSplitPass begin----------" << std::endl;
    std::map<std::string, std::map<int32_t, std::vector<GNode>>> matmul_node_map;
    FindSameLevelMatmulNode(graph, custom_context, matmul_node_map);
    int32_t affect_count = 0;
    auto ret = ReplaceMatmul(graph, custom_context, matmul_node_map, affect_count);
    std::cout << "FusePackBatchMatmulSplitPass end, affect count is: " << affect_count << std::endl;
    return ret;
}

REGISTER_CUSTOM_PASS("FusePackBatchMatmulSplitPass")
    .CustomPassFn(FusePackBatchMatmulSplitPass)
    .Stage(CustomPassStage::kAfterInferShape);
