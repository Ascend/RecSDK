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
#include "ge_ir_build.h"
#include "all_ops.h"

using namespace ge;

namespace {
constexpr const char* const kTypeMatmulV2 = "MatMulV2";
constexpr const char* const kTypeConst = "Const";
constexpr const char* const kTypeConstant = "Constant";
const size_t FOO_MAX_LEN = 1024 * 1024 * 1024;
}  // namespace

void FindNodesCanFusion(GraphPtr& graph, std::vector<GNode>& mm_nodes)
{
    for (const GNode& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == kTypeMatmulV2) {
            mm_nodes.emplace_back(node);
        }
    }
}

size_t GetDataSize(DataType data_type)
{
    if (data_type == DT_FLOAT || data_type == DT_INT32) {
        return 4;
    }

    if (data_type == DT_INT16 || data_type == DT_FLOAT16) {
        return 2;
    }

    return 0;
}

template <typename T>
void TransposeMatrix(uint8_t* input_data, uint8_t* output_data, size_t rows, size_t cols)
{
    const T* input = reinterpret_cast<const T*>(input_data);
    T* output = reinterpret_cast<T*>(output_data);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

bool IsMiniNPattern(size_t k_dim, size_t n_dim, size_t type_size, bool transpose_b)
{
    return k_dim >= 1000 && n_dim < 32 && n_dim > 1 && n_dim * type_size % 32 != 0 && !transpose_b;
}

bool IsAlignNPattern(size_t k_dim, size_t n_dim, size_t type_size, bool transpose_b)
{
    if (!transpose_b) {
        return false;
    }

    if (n_dim * type_size % 8 != 0) {
        return false;
    }

    if (k_dim < 1600 || k_dim >= 6000) {
        return false;
    }

    if (k_dim >= 1600 && k_dim <= 2200 && n_dim != 64 && n_dim != 128 && n_dim != 256) {
        return false;
    }

    if (n_dim != 128 && n_dim != 256 && (n_dim < 24 || n_dim > 64)) {
        return false;
    }

    return true;
}

graphStatus ReplaceMmNodes(GraphPtr& graph, GNode& mm_node, CustomPassContext& context, int64_t& replaced_count)
{
    AscendString mm_name;
    if (mm_node.GetName(mm_name) != GRAPH_SUCCESS) {
        std::cout << "mm node get name failed" << std::endl;
        return GRAPH_FAILED;
    }
    std::string new_mm_name = std::string(mm_name.GetString());

    TensorDesc mm_input_desc_0;
    Shape mm_input_0_shape;
    if (mm_node.GetInputDesc(0, mm_input_desc_0) != GRAPH_SUCCESS) {
        std::cout << "mm node get input0 desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    mm_input_0_shape = mm_input_desc_0.GetShape();
    std::vector<int64_t> mm_input_0_dims = mm_input_0_shape.GetDims();

    TensorDesc mm_input_desc_1;
    Shape mm_input_1_shape;
    Shape mm_input_1_ori_shape;
    if (mm_node.GetInputDesc(1, mm_input_desc_1) != GRAPH_SUCCESS) {
        std::cout << "mm node get input1 desc failed" << std::endl;
        return GRAPH_FAILED;
    }
    mm_input_1_shape = mm_input_desc_1.GetShape();
    mm_input_1_ori_shape = mm_input_desc_1.GetOriginShape();
    std::vector<int64_t> mm_input_1_dims = mm_input_1_shape.GetDims();

    DataType mm_data_type = mm_input_desc_0.GetDataType();
    size_t data_size = GetDataSize(mm_data_type);
    if (data_size == 0) {
        return GRAPH_SUCCESS;
    }

    bool transpose_b;
    if (mm_node.GetAttr(AscendString("transpose_x2"), transpose_b) != GRAPH_SUCCESS) {
        std::cout << "mm node get adj_x2 attr failed" << std::endl;
        return GRAPH_FAILED;
    }

    size_t k_dim = mm_input_1_dims[0];
    size_t n_dim = mm_input_1_dims[1];
    if (transpose_b) {
        n_dim = mm_input_1_dims[0];
        k_dim = mm_input_1_dims[1];
    }

    if (!IsMiniNPattern(k_dim, n_dim, data_size, transpose_b) &&
        !IsAlignNPattern(k_dim, n_dim, data_size, transpose_b)) {
        return GRAPH_SUCCESS;
    }

    auto mm_input_node1 = mm_node.GetInDataNodesAndPortIndexs(1).first;
    int32_t mm_input_1_index = mm_node.GetInDataNodesAndPortIndexs(1).second;
    AscendString mm_input_1_type;
    mm_input_node1->GetType(mm_input_1_type);
    if (mm_input_1_type != kTypeConst && mm_input_1_type != kTypeConstant) {
        return GRAPH_SUCCESS;
    }

    Tensor input_1_data_tensor;
    if (mm_node.GetInputConstData(1, input_1_data_tensor) != GRAPH_SUCCESS) {
        std::cout << "get input1 data failed" << std::endl;
        return GRAPH_FAILED;
    }
    DataType input_1_data_type = input_1_data_tensor.GetDataType();
    size_t input_1_size = input_1_data_tensor.GetSize();
    uint8_t* input_1_data = input_1_data_tensor.GetData();
    if (input_1_size < 0 || input_1_size > FOO_MAX_LEN) {
        return GRAPH_SUCCESS;
    }
    uint8_t* input_1_data_trans = new uint8_t[input_1_size];
    TensorDesc input_1_tensor_desc = input_1_data_tensor.GetTensorDesc();

    Shape shape = input_1_tensor_desc.GetShape();
    if (shape.GetDimNum() < 2) {
        std::cout << "unsupport const input shape" << std::endl;
        return GRAPH_FAILED;
    }
    size_t rows = shape.GetDim(0);
    size_t cols = shape.GetDim(1);

    if (input_1_data_type == DT_FLOAT) {
        TransposeMatrix<float>(input_1_data, input_1_data_trans, rows, cols);
    } else {
        std::cout << "unsupport weight dtype" << std::endl;
        return GRAPH_SUCCESS;
    }

    transpose_b = !transpose_b;
    if (mm_node.SetAttr(AscendString("transpose_x2"), transpose_b) != GRAPH_SUCCESS) {
        std::cout << "mm node set adj_x2 attr failed" << std::endl;
        return GRAPH_FAILED;
    }
    if (transpose_b) {
        mm_input_1_shape.SetDim(0, n_dim);
        mm_input_1_shape.SetDim(1, k_dim);
        mm_input_1_ori_shape.SetDim(0, n_dim);
        mm_input_1_ori_shape.SetDim(1, k_dim);
    } else {
        mm_input_1_shape.SetDim(0, k_dim);
        mm_input_1_shape.SetDim(1, n_dim);
        mm_input_1_ori_shape.SetDim(0, k_dim);
        mm_input_1_ori_shape.SetDim(1, n_dim);
    }

    mm_input_desc_1.SetShape(mm_input_1_shape);
    mm_input_desc_1.SetOriginShape(mm_input_1_ori_shape);
    if (mm_node.UpdateInputDesc(1, mm_input_desc_1) != GRAPH_SUCCESS) {
        std::cout << "matmul node UpdateInputDesc 1 failed" << std::endl;
        return GRAPH_FAILED;
    }

    AscendString mm_input_name1;
    if (mm_input_node1->GetName(mm_input_name1) != GRAPH_SUCCESS) {
        std::cout << "mm input 1 get name failed" << std::endl;
        return GRAPH_FAILED;
    }
    std::string mm_input_name1_str = std::string(mm_input_name1.GetString());
    std::string new_mm_input_name1_str = mm_input_name1_str + "/trans";
    auto new_weight = op::Const(new_mm_input_name1_str.c_str());
    TensorDesc new_weight_desc(ge::Shape({cols, rows}), FORMAT_ND, input_1_data_type);
    new_weight_desc.SetOriginShape(ge::Shape({cols, rows}));
    Tensor new_weight_tensor(new_weight_desc, input_1_data_trans, input_1_size);
    delete[] input_1_data_trans;
    new_weight.set_attr_value(new_weight_tensor);
    GNode new_weight_gnode = graph->AddNodeByOp(new_weight);
    TensorDesc new_weight_gnode_desc(new_weight_desc);
    if (new_weight_gnode.UpdateOutputDesc(0, new_weight_gnode_desc) != GRAPH_SUCCESS) {
        std::cout << "update new weight op desc failed" << std::endl;
        return GRAPH_FAILED;
    }

    auto mm_input_node1_outputs = mm_input_node1->GetOutDataNodesAndPortIndexs(0);
    if (mm_input_node1_outputs.size() == 1) {
        if (graph->RemoveNode(*mm_input_node1) != GRAPH_SUCCESS) {
            std::cout << "remove ori weight failed" << std::endl;
            return GRAPH_FAILED;
        }
    } else {
        if (graph->RemoveEdge(*mm_input_node1, 0, mm_node, 1) != GRAPH_SUCCESS) {
            std::cout << "remove mm const edge failed" << std::endl;
            return GRAPH_FAILED;
        }
    }
    if (graph->AddDataEdge(new_weight_gnode, 0, mm_node, 1) != GRAPH_SUCCESS) {
        std::cout << "add edge new weight to mm failed" << std::endl;
        return GRAPH_FAILED;
    }

    replaced_count++;

    return GRAPH_SUCCESS;
}

graphStatus MatmulTransposeConst(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "----------Matmul Transpose Const Pass begin----------" << std::endl;
    std::vector<GNode> mm_nodes;
    FindNodesCanFusion(graph, mm_nodes);

    if (mm_nodes.empty()) {
        std::cout << "Not found mm node" << std::endl;
        return GRAPH_SUCCESS;
    }

    int64_t replaced_count = 0;

    for (GNode& node : mm_nodes) {
        if (ReplaceMmNodes(graph, node, context, replaced_count) != GRAPH_SUCCESS) {
            std::cout << "replace mm_nodes failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    std::cout << "Matmul Transpose Const Pass end, affect count: " << replaced_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("MatmulTransposeConstPass")
    .CustomPassFn(MatmulTransposeConst)
    .Stage(CustomPassStage::kAfterInferShape);
