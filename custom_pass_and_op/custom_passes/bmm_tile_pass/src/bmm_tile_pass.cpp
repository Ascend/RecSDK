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
constexpr const char* const kTypeReshape = "Reshape";
constexpr const char* const kTypeBatchMatmul = "BatchMatMulV2";
constexpr const char* const kTypeTranspose = "Transpose";
constexpr const char* const kTypeTile = "Tile";
constexpr const char* const kTypeConst = "Const";
constexpr const char* const kTypeConstant = "Constant";
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

bool CheckNodeOutputsNum(std::shared_ptr<GNode> tile_node)
{
    auto outputs = tile_node->GetOutDataNodesAndPortIndexs(0);
    return outputs.size() == 1;
}

graphStatus DeleteTileOp(GraphPtr& graph, GNode& bmm_node, GNode& tile_node, int32_t tile_node_index,
                         CustomPassContext& context)
{
    ge::AscendString tile_name;
    tile_node.GetName(tile_name);
    auto tile_input = tile_node.GetInDataNodesAndPortIndexs(0);
    TensorDesc tile_input_desc;
    if (tile_node.GetInputDesc(0, tile_input_desc) != GRAPH_SUCCESS) {
        std::cout << "get tile node input 0 failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (graph->RemoveNode(tile_node) != GRAPH_SUCCESS) {
        std::cout << "remove tile node failed" << std::endl;
        return GRAPH_FAILED;
    };
    ge::AscendString tile_input_name;
    bmm_node.GetName(tile_input_name);
    bmm_node.UpdateInputDesc(tile_node_index, tile_input_desc);
    if (graph->AddDataEdge(*tile_input.first, tile_input.second, bmm_node, tile_node_index) != GRAPH_SUCCESS) {
        std::cout << "add edge tile input to bmm failed" << std::endl;
        return GRAPH_FAILED;
    };
    return GRAPH_SUCCESS;
}

graphStatus CheckBmmTileShape(GNode& bmm_node, GNode& tile_node, int32_t index)
{
    TensorDesc bmm_desc;
    TensorDesc tile_input_desc;
    TensorDesc tile_output_desc;

    if (bmm_node.GetInputDesc(index, bmm_desc) != GRAPH_SUCCESS) {
        std::cout << "get bmm input desc failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (tile_node.GetInputDesc(0, tile_input_desc) != GRAPH_SUCCESS) {
        std::cout << "get tile input desc 0 failed" << std::endl;
        return GRAPH_FAILED;
    };
    if (tile_node.GetOutputDesc(0, tile_output_desc) != GRAPH_SUCCESS) {
        std::cout << "get tile output desc 0 failed" << std::endl;
        return GRAPH_FAILED;
    };

    Shape bmm_shape;
    Shape tile_input_shape;
    Shape tile_output_shape;

    bmm_shape = bmm_desc.GetShape();
    tile_input_shape = tile_input_desc.GetShape();
    tile_output_shape = tile_output_desc.GetShape();

    std::vector<int64_t> bmm_dims;
    std::vector<int64_t> tile_input_dims;
    std::vector<int64_t> tile_output_dims;

    bmm_dims = bmm_shape.GetDims();
    tile_input_dims = tile_input_shape.GetDims();
    tile_output_dims = tile_output_shape.GetDims();

    for (int32_t i = 0; i < tile_input_shape.GetDimNum(); i++) {
        if (i == 0) {
            if (tile_input_dims[i] != 1 || bmm_dims[i] == 1) {
                return GRAPH_FAILED;
            }
        } else {
            if (tile_output_dims[i] != tile_input_dims[i]) {
                return GRAPH_FAILED;
            }
        }
    }

    return GRAPH_SUCCESS;
}

graphStatus ReplaceSingleTile(GraphPtr& graph, GNode& node, CustomPassContext& context, int64_t& replaced_count)
{
    auto bmm_input_node0 = node.GetInDataNodesAndPortIndexs(0).first;
    AscendString bmm_input_type0;
    bmm_input_node0->GetType(bmm_input_type0);

    auto bmm_input_node1 = node.GetInDataNodesAndPortIndexs(1).first;
    AscendString bmm_input_type1;
    bmm_input_node1->GetType(bmm_input_type1);

    if (bmm_input_type1 != kTypeTile && bmm_input_type0 != kTypeTile) {
        return GRAPH_SUCCESS;
    }

    if (bmm_input_type1 == kTypeTile && bmm_input_type0 == kTypeTile) {
        return GRAPH_SUCCESS;
    }

    if (bmm_input_type0 == kTypeTile && CheckNodeOutputsNum(bmm_input_node0) &&
        CheckBmmTileShape(node, *bmm_input_node0, 1) == GRAPH_SUCCESS) {
        if (DeleteTileOp(graph, node, *bmm_input_node0, 0, context) != GRAPH_SUCCESS) {
            std::cout << "delte tile op failed" << std::endl;
            return GRAPH_FAILED;
        };
        replaced_count += 1;
    }

    if (bmm_input_type1 == kTypeTile && CheckNodeOutputsNum(bmm_input_node1) &&
        CheckBmmTileShape(node, *bmm_input_node1, 0) == GRAPH_SUCCESS) {
        if (DeleteTileOp(graph, node, *bmm_input_node1, 1, context) != GRAPH_SUCCESS) {
            std::cout << "delete tile op failed" << std::endl;
            return GRAPH_FAILED;
        };
        replaced_count += 1;
    }

    return GRAPH_SUCCESS;
}

bool GetInt32SizeAndData(std::shared_ptr<GNode> base_node, int32_t idx, size_t& size, int32_t*& int32_data,
                         Tensor& const_tensor)
{
    if (base_node->GetInputConstData(idx, const_tensor) != GRAPH_SUCCESS) {
        return false;
    }

    if (const_tensor.GetDataType() != DT_INT32) {
        return false;
    }

    size_t const_size = const_tensor.GetSize();
    uint8_t* const_data = const_tensor.GetData();

    if (const_size == 0 || const_data == nullptr || const_size % 4 != 0) {
        return false;
    }

    size = const_size / 4;
    int32_data = reinterpret_cast<int32_t*>(const_data);

    return true;
}

bool CheckTransposePerm(std::shared_ptr<GNode> transpose_node)
{
    Tensor const_tensor;
    size_t perm_size;
    int32_t* int32_data;

    if (!GetInt32SizeAndData(transpose_node, 1, perm_size, int32_data, const_tensor)) {
        return false;
    }

    // dim 0 has been transposed
    if (int32_data[0] != 0) {
        return false;
    }

    return true;
}

bool GetAxisFirstDim(std::shared_ptr<GNode> reshape_node, int32_t& dim0)
{
    Tensor const_tensor;
    size_t axis_size;
    int32_t* int32_data;

    if (!GetInt32SizeAndData(reshape_node, 1, axis_size, int32_data, const_tensor)) {
        return false;
    }

    dim0 = int32_data[0];

    return true;
}

bool CheckTileMultiples(std::shared_ptr<GNode> tile_node, int32_t dim0_multiples)
{
    Tensor const_tensor;
    size_t multiples_size;
    int32_t* int32_data;

    if (!GetInt32SizeAndData(tile_node, 1, multiples_size, int32_data, const_tensor)) {
        return false;
    }

    if (int32_data[0] != dim0_multiples) {
        return false;
    }

    for (int32_t i = 1; i < multiples_size; i++) {
        if (int32_data[i] != 1) {
            return false;
        }
    }

    return true;
}

graphStatus UpdateReshapeAxis(std::shared_ptr<GNode> reshape_node)
{
    Tensor const_tensor;
    size_t axis_size;
    int32_t* int32_data;

    if (!GetInt32SizeAndData(reshape_node, 1, axis_size, int32_data, const_tensor)) {
        std::cout << "GetReshape Axis failed" << std::endl;
        return GRAPH_FAILED;
    }

    int32_data[0] = 0;

    return const_tensor.SetData(reinterpret_cast<uint8_t*>(int32_data), axis_size * 4);
}

graphStatus ReplaceTileTranspose(GraphPtr& graph, GNode& node, CustomPassContext& context, int64_t& replaced_count)
{
    auto bmm_input_node0 = node.GetInDataNodesAndPortIndexs(0).first;
    AscendString bmm_input_type0;
    bmm_input_node0->GetType(bmm_input_type0);

    auto bmm_input_node1 = node.GetInDataNodesAndPortIndexs(1).first;
    AscendString bmm_input_type1;
    bmm_input_node1->GetType(bmm_input_type1);

    if (bmm_input_type1 != kTypeTranspose && bmm_input_type0 != kTypeTranspose) {
        return GRAPH_SUCCESS;
    }

    // only deal with input1 when both input nodes are transpose
    std::shared_ptr<GNode> transpose_node;
    int32_t transpose_idx = 0;
    if (bmm_input_type0 == kTypeTranspose) {
        transpose_node = bmm_input_node0;
    }

    if (bmm_input_type1 == kTypeTranspose) {
        transpose_node = bmm_input_node1;
        transpose_idx = 1;
    }

    auto reshape_node = transpose_node->GetInDataNodesAndPortIndexs(0).first;
    AscendString reshape_type;
    reshape_node->GetType(reshape_type);

    auto transpose_perm_node = transpose_node->GetInDataNodesAndPortIndexs(1).first;
    AscendString transpose_perm_type;
    transpose_perm_node->GetType(transpose_perm_type);

    if (reshape_type != kTypeReshape) {
        return GRAPH_SUCCESS;
    }

    if (transpose_perm_type != kTypeConst && transpose_perm_type != kTypeConstant) {
        return GRAPH_SUCCESS;
    }

    if (!CheckTransposePerm(transpose_node)) {
        return GRAPH_SUCCESS;
    }

    auto tile_node = reshape_node->GetInDataNodesAndPortIndexs(0).first;
    AscendString tile_type;
    tile_node->GetType(tile_type);

    auto reshape_axis_node = reshape_node->GetInDataNodesAndPortIndexs(1).first;
    AscendString axis_type;
    reshape_axis_node->GetType(axis_type);

    if (tile_type != kTypeTile) {
        return GRAPH_SUCCESS;
    }

    if (axis_type != kTypeConst && axis_type != kTypeConstant) {
        return GRAPH_SUCCESS;
    }

    int32_t axis_first_dim;
    if (!GetAxisFirstDim(reshape_node, axis_first_dim)) {
        return GRAPH_SUCCESS;
    }

    if (!CheckTileMultiples(tile_node, axis_first_dim)) {
        return GRAPH_SUCCESS;
    }

    if (!CheckNodeOutputsNum(tile_node) || !CheckNodeOutputsNum(reshape_node) || !CheckNodeOutputsNum(transpose_node)) {
        return GRAPH_SUCCESS;
    }

    if (CheckBmmTileShape(node, *tile_node, 1 - transpose_idx) != GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
    }

    auto tile_input = tile_node->GetInDataNodesAndPortIndexs(0);
    TensorDesc tile_input_desc;
    if (tile_input.first->GetOutputDesc(0, tile_input_desc) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    };

    if (graph->RemoveNode(*tile_node) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    };
    if (graph->AddDataEdge(*tile_input.first, tile_input.second, *reshape_node, 0) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    };
    if (UpdateReshapeAxis(reshape_node) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    };

    TensorDesc reshape_input_desc(tile_input_desc);
    if (reshape_node->UpdateInputDesc(0, reshape_input_desc))
        ;

    TensorDesc reshape_output_desc;
    if (reshape_node->GetOutputDesc(0, reshape_output_desc) != GRAPH_SUCCESS) {
        std::cout << "reshape node get output desc failed" << std::endl;
        return GRAPH_FAILED;
    };
    Shape reshape_output_shape = reshape_output_desc.GetShape();
    reshape_output_shape.SetDim(0, 1);
    reshape_output_desc.SetShape(reshape_output_shape);
    reshape_output_desc.SetOriginShape(reshape_output_shape);
    if (reshape_node->UpdateOutputDesc(0, reshape_output_desc) != GRAPH_SUCCESS) {
        std::cout << "update reshape node output desc failed" << std::endl;
        return GRAPH_FAILED;
    };

    TensorDesc transpose_input_desc;
    if (transpose_node->GetInputDesc(0, transpose_input_desc) != GRAPH_SUCCESS) {
        std::cout << "get transpose node input desc failed" << std::endl;
        return GRAPH_FAILED;
    };
    Shape transpose_input_shape = transpose_input_desc.GetShape();
    transpose_input_shape.SetDim(0, 1);
    transpose_input_desc.SetShape(transpose_input_shape);
    transpose_input_desc.SetOriginShape(transpose_input_shape);
    if (transpose_node->UpdateInputDesc(0, transpose_input_desc) != GRAPH_SUCCESS) {
        std::cout << "transpose node update input desc failed" << std::endl;
        return GRAPH_FAILED;
    };

    TensorDesc transpose_output_desc;
    if (transpose_node->GetOutputDesc(0, transpose_output_desc) != GRAPH_SUCCESS) {
        std::cout << "transpose node get ouput desc failed" << std::endl;
        return GRAPH_FAILED;
    };
    Shape transpose_output_shape = transpose_output_desc.GetShape();
    transpose_output_shape.SetDim(0, 1);
    transpose_output_desc.SetShape(transpose_output_shape);
    transpose_output_desc.SetOriginShape(transpose_output_shape);
    if (transpose_node->UpdateOutputDesc(0, transpose_output_desc) != GRAPH_SUCCESS) {
        std::cout << "tranpose node update output desc failed" << std::endl;
        return GRAPH_FAILED;
    };

    TensorDesc bmm_new_input_desc(transpose_output_desc);
    if (node.UpdateInputDesc(transpose_idx, bmm_new_input_desc) != GRAPH_SUCCESS) {
        std::cout << "bmm node update input desc failed" << std::endl;
        return GRAPH_FAILED;
    };

    replaced_count++;

    return GRAPH_SUCCESS;
}

graphStatus BatchMatmulTileCustomPass(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------BatchMatmul_Tile_Pass begin----------" << std::endl;
    std::vector<GNode> node_list;
    FindNodesCanFusion(graph, node_list);

    if (node_list.empty()) {
        std::cout << "Not found BatchMatmul_Tile_CustomPass node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;
    for (GNode& node : node_list) {
        if (ReplaceSingleTile(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        };
        if (ReplaceTileTranspose(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        };
    }
    std::cout << "BatchMatmulTilePass end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("BatchMatmulTilePass")
    .CustomPassFn(BatchMatmulTileCustomPass)
    .Stage(CustomPassStage::kAfterInferShape);
