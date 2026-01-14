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
#include "register_custom_pass.h"
#include "all_ops.h"

using namespace std;
using namespace ge;

namespace {

constexpr const char* kEinSum = "Einsum";
constexpr const char* kReshape = "Reshape";
constexpr const char* kInputShape = "shape";
constexpr const char* kInputPerm = "perm";
constexpr const char* kTranspose = "Transpose";
constexpr const char* kTransposeD = "TransposeD";
constexpr const char* kMatmulV2 = "MatMulV2";
constexpr const char* kBatchMatmulV2 = "BatchMatMulV2";
constexpr const char* kAttrTransposeX1 = "transpose_x1";
constexpr const char* kAttrTransposeX2 = "transpose_x2";
constexpr const char* kAttrAdjX1 = "adj_x1";
constexpr const char* kAttrAdjX2 = "adj_x2";
constexpr const char* kAttrPerm = "perm";
constexpr const char* kAttrEquation = "equation";
static constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
static constexpr uint8_t TOTAL_LABELS = NUM_OF_LETTERS * 2;
static constexpr uint8_t ELLIPSIS = TOTAL_LABELS;

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

enum class EinsumDimensionType {
    REDUCE = 0,
    FREE = 1,
    SUM_K = 2,
    BATCH = 3,
    ONLY_OUTPUT = 4
};

struct LabelInfo {
    std::vector<uint8_t> labels;
    std::vector<int32_t> indices;
};

enum class EinSumPassStatus {
    SUCCESS = 0,
    NOT_SUPPORT = 1,
    ERROR = 2
};

static uint8_t label_to_subscript(unsigned char label)
{
    return std::isupper(label) ? label - 'A' : label - 'a' + NUM_OF_LETTERS;
}

static unsigned char subscript_to_label(uint8_t s)
{
    return s < NUM_OF_LETTERS ? s + 'A' : s + 'a' - NUM_OF_LETTERS;
}

template <typename T>
bool CompareTwoVector(const std::vector<T>& vector_0, const std::vector<T>& vector_1)
{
    auto size_0 = vector_0.size();
    auto size_1 = vector_1.size();
    if (size_0 != size_1) {
        return true;
    } else {
        for (size_t i = 0; i < size_0; ++i) {
            if (vector_0[i] != vector_1[i]) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

EinSumPassStatus TransLabels(const std::string& child_equation, const size_t& num_ops,
                             std::vector<std::vector<uint8_t>>& op_labels, std::vector<uint8_t>& label_used)
{
    constexpr std::size_t ELLIPSIS_LAST_CHAR_OFFSET = 2;
    std::size_t curr_op = 0;
    bool ell_in_input = false;
    size_t ell_loc = 0;
    for (std::size_t i = 0; i < child_equation.length(); ++i) {
        if (ell_in_input && ell_loc + ELLIPSIS_LAST_CHAR_OFFSET >= i) {
            continue;
        }
        const unsigned char label = child_equation[i];
        switch (label) {
            case ' ':
                break;
            case '.':
                if (ell_in_input) {
                    std::cout << "einsum found \'.\' for operand " + to_string(curr_op) +
                                     "for which an ellipsis was already found."
                              << std::endl;
                    return EinSumPassStatus::ERROR;
                }
                if (!(i + ELLIPSIS_LAST_CHAR_OFFSET < child_equation.length() && child_equation[i + 1] == '.' &&
                      child_equation[i + 2] == '.')) {
                    std::cout << "einsum found \'.\' for operand " + to_string(curr_op) +
                                     "that is not part of any ellipsis."
                              << std::endl;
                    return EinSumPassStatus::ERROR;
                }
                op_labels[curr_op].push_back(ELLIPSIS);
                ell_in_input = true;
                ell_loc = i;
                break;
            case ',':
                ++curr_op;
                if (curr_op >= num_ops) {
                    std::cout << "einsum fewer operands were provided than specified in the equation num_ops."
                              << std::endl;
                    return EinSumPassStatus::ERROR;
                }
                ell_in_input = false;
                break;
            default:
                if (!std::isalpha(label)) {
                    std::cout << "einsum invalid subscript given at index" + to_string(i) +
                                     "in the equation string, subscripts must be in [a-zA-Z]."
                              << std::endl;
                    return EinSumPassStatus::ERROR;
                }

                uint8_t subscript = label_to_subscript(label);
                label_used[subscript] = 1;
                op_labels[curr_op].push_back(subscript);
        }
    }
    if (curr_op != num_ops - 1) {
        std::cout << "einsum more operands were provided than specified in the equation." << std::endl;
        return EinSumPassStatus::ERROR;
    }
    return EinSumPassStatus::SUCCESS;
}

EinSumPassStatus SplitEquation(const size_t& num_ops, const size_t& output_num, const std::string& equation,
                               std::vector<std::vector<uint8_t>>& op_labels,
                               std::vector<std::vector<uint8_t>>& out_labels, std::vector<uint8_t>& label_used)
{
    const auto arrow_pos = equation.find("->");
    const auto lhs = equation.substr(0, arrow_pos);
    const auto rhs = equation.substr(arrow_pos + 2);
    if (TransLabels(lhs, num_ops, op_labels, label_used) != EinSumPassStatus::SUCCESS) {
        std::cout << "trans labels for input failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }
    if (TransLabels(rhs, output_num, out_labels, label_used) != EinSumPassStatus::SUCCESS) {
        std::cout << "trans labels for output failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }
    return EinSumPassStatus::SUCCESS;
}

EinSumPassStatus GetOneNewSubscript(std::vector<uint8_t>& label_used, uint8_t& new_label)
{
    for (size_t i = 0; i < label_used.size(); ++i) {
        if (label_used[i] == 0) {
            label_used[i] = 1;
            new_label = i;
            return EinSumPassStatus::SUCCESS;
        }
    }
    return EinSumPassStatus::ERROR;
}

EinSumPassStatus EllipsisReplace(std::vector<TensorDesc>& op_descs, std::vector<std::vector<uint8_t>>& op_labels,
                                 std::vector<uint8_t>& out_labels, std::vector<uint8_t>& label_used)
{
    int64_t ell_num_dim = 0;
    std::vector<uint8_t> ell_subscripts;
    for (size_t i = 0; i < op_labels.size(); ++i) {
        TensorDesc op_desc = op_descs[i];
        std::vector<uint8_t>& labels = op_labels[i];
        auto n_labels = labels.size();
        auto op_shape = op_desc.GetShape();
        auto dims = op_shape.GetDimNum();
        auto ndims = dims - (n_labels - 1);
        auto iter = std::find(labels.begin(), labels.end(), ELLIPSIS);
        if (iter != labels.end()) {
            auto ell_index = distance(labels.begin(), iter);
            if (ell_num_dim != 0 && ell_num_dim != ndims) {
                std::cout << "only support two inputs with the same number of axes omitted!" << std::endl;
                return EinSumPassStatus::NOT_SUPPORT;
            } else if (ell_num_dim == 0) {
                ell_num_dim = ndims;
                for (int64_t k = 0; k < ell_num_dim; k++) {
                    uint8_t new_subscript;
                    if (GetOneNewSubscript(label_used, new_subscript) != EinSumPassStatus::SUCCESS) {
                        std::cout << "no more subcript available" << std::endl;
                        return EinSumPassStatus::ERROR;
                    };
                    ell_subscripts.push_back(new_subscript);
                }
            }
            labels.erase(labels.begin() + ell_index);
            labels.insert(labels.begin() + ell_index, ell_subscripts.begin(), ell_subscripts.end());
        }
        if (iter == labels.end()) {
            continue;
        }
        auto ell_index = distance(labels.begin(), iter);
        if (ell_num_dim != 0 && ell_num_dim != ndims) {
            std::cout << "only support two inputs with the same number of axes omitted!" << std::endl;
            return EinSumPassStatus::NOT_SUPPORT;
        }
        if (ell_num_dim != 0) {
            labels.erase(labels.begin() + ell_index);
            labels.insert(labels.begin() + ell_index, ell_subscripts.begin(), ell_subscripts.end());
            continue;
        }
        ell_num_dim = ndims;
        for (int64_t k = 0; k < ell_num_dim; k++) {
            uint8_t new_subscript;
            if (GetOneNewSubscript(label_used, new_subscript) != EinSumPassStatus::SUCCESS) {
                std::cout << "no more subcript available" << std::endl;
                return EinSumPassStatus::ERROR;
            };
            ell_subscripts.push_back(new_subscript);
        }
        labels.erase(labels.begin() + ell_index);
        labels.insert(labels.begin() + ell_index, ell_subscripts.begin(), ell_subscripts.end());
    }
    //  输出替换
    auto iter = std::find(out_labels.begin(), out_labels.end(), ELLIPSIS);
    if (iter != out_labels.end()) {
        auto ell_out_index = distance(out_labels.begin(), iter);
        out_labels.erase(out_labels.begin() + ell_out_index);
        out_labels.insert(out_labels.begin() + ell_out_index, ell_subscripts.begin(), ell_subscripts.end());
    }
    return EinSumPassStatus::SUCCESS;
}

void LabelCountMap(std::vector<std::vector<uint8_t>>& op_labels, std::vector<uint8_t>& label_count)
{
    for (size_t i = 0; i < op_labels.size(); i++) {
        std::vector<uint8_t> op_label = op_labels[i];
        for (auto& label : op_label) {
            ++label_count[label];
        }
    }
}

void GetDimsType(std::vector<uint8_t>& label_count, std::vector<uint8_t>& output_labels,
                 std::map<uint8_t, EinsumDimensionType>& dims_type_map)
{
    for (size_t i = 0; i < TOTAL_LABELS; ++i) {
        auto count = label_count[i];
        bool exist_in_output = find(output_labels.begin(), output_labels.end(), i) != output_labels.end();
        if (count == 1) {
            if (exist_in_output) {
                dims_type_map[i] = EinsumDimensionType::FREE;
            } else {
                dims_type_map[i] = EinsumDimensionType::REDUCE;
            }
        } else if (count == 2) {
            if (exist_in_output) {
                dims_type_map[i] = EinsumDimensionType::BATCH;
            } else {
                dims_type_map[i] = EinsumDimensionType::SUM_K;
            }
        } else if (count == 0 && exist_in_output) {
            dims_type_map[i] = EinsumDimensionType::ONLY_OUTPUT;
        }
    }
}

void CollectDimensionType(std::vector<std::vector<uint8_t>>& labels,
                          std::map<uint8_t, EinsumDimensionType>& dims_type_map,
                          std::vector<std::map<EinsumDimensionType, LabelInfo>>& label_infos)
{
    for (size_t i = 0; i < labels.size(); ++i) {
        for (size_t j = 0; j < labels[i].size(); ++j) {
            auto label = labels[i][j];
            EinsumDimensionType dim_type = dims_type_map[label];
            label_infos[i][dim_type].labels.push_back(label);
            label_infos[i][dim_type].indices.push_back(static_cast<int32_t>(j));
        }
    }
}

EinSumPassStatus BatchRevise(std::map<uint8_t, EinsumDimensionType>& dims_type_map,
                             std::vector<std::vector<uint8_t>>& input_labels, std::vector<TensorDesc>& input_descs,
                             std::vector<uint8_t>& label_used)
{
    for (const auto& pair : dims_type_map) {
        uint8_t label = pair.first;
        EinsumDimensionType dim_type = pair.second;
        auto index_left_it = find(input_labels[0].begin(), input_labels[0].end(), label);
        auto index_right_it = find(input_labels[1].begin(), input_labels[1].end(), label);
        int32_t dim_left = 0;
        int32_t dim_right = 0;
        int32_t index_left, index_right;
        if (index_left_it != input_labels[0].end()) {
            index_left = distance(input_labels[0].begin(), index_left_it);
            dim_left = input_descs[0].GetShape().GetDim(index_left);
        }
        if (index_right_it != input_labels[1].end()) {
            index_right = distance(input_labels[1].begin(), index_right_it);
            dim_right = input_descs[1].GetShape().GetDim(index_right);
        }
        if (dim_type == EinsumDimensionType::SUM_K) {
            if (dim_left != dim_right) {
                std::cout << "label " + std::string(1, subscript_to_label(label)) +
                                 " is k but size in left and right not equal."
                          << std::endl;
                return EinSumPassStatus::NOT_SUPPORT;
            }
        } else if (dim_type == EinsumDimensionType::BATCH) {
            if (dim_left != 1 && dim_right != 1 && dim_left != dim_right) {
                std::cout << "label " + std::string(1, subscript_to_label(label)) +
                                 " does not meet the broadcast condition."
                          << std::endl;
                return EinSumPassStatus::ERROR;
            }
            uint8_t new_subscript;
            if (GetOneNewSubscript(label_used, new_subscript) != EinSumPassStatus::SUCCESS) {
                std::cout << "no more subcript success" << std::endl;
                return EinSumPassStatus::ERROR;
            };
            if (dim_left == 1 && dim_right != 1) {
                input_labels[0][index_left] = new_subscript;
                dims_type_map[new_subscript] = EinsumDimensionType::REDUCE;
                dims_type_map[label] = EinsumDimensionType::FREE;
            }
        } else if (dim_type == EinsumDimensionType::ONLY_OUTPUT) {
            return EinSumPassStatus::ERROR;
        }
    }
    return EinSumPassStatus::SUCCESS;
}

void GetDimSize(TensorDesc& op_desc, std::vector<int64_t>& dim_sizes)
{
    auto dim_num = op_desc.GetShape().GetDimNum();
    for (size_t i = 0; i < dim_num; ++i) {
        dim_sizes.push_back(op_desc.GetShape().GetDim(i));
    }
}

EinSumPassStatus CheckDimType(std::vector<std::vector<uint8_t>>& ori_labels, std::vector<GNode>& input_nodes,
                              std::vector<int32_t> input_out_indexs,
                              std::vector<std::map<EinsumDimensionType, LabelInfo>>& label_infos,
                              std::map<uint8_t, EinsumDimensionType>& dims_type_map,
                              std::vector<std::vector<uint8_t>>& new_labels, std::vector<uint8_t>& label_used)
{
    for (size_t i = 0; i < ori_labels.size(); ++i) {
        GNode input_node = input_nodes[i];
        int32_t input_out_index = input_out_indexs[i];
        auto label_info = label_infos[i];
        std::vector<uint8_t>& new_label = new_labels[i];
        std::vector<uint8_t> ori_label = ori_labels[i];
        // 获取op_desc
        TensorDesc op_desc;
        if (input_node.GetOutputDesc(input_out_index, op_desc) != GRAPH_SUCCESS) {
            std::cout << "get input node out desc failed." << std::endl;
            return EinSumPassStatus::ERROR;
        }
        auto input_shape = op_desc.GetShape();
        auto dim_num = input_shape.GetDimNum();
        std::vector<bool> remove_label(dim_num, false);
        // 多个K轴跳过
        auto it = label_info.find(EinsumDimensionType::SUM_K);
        if (it != label_info.end() && (it->second.labels.size() > 1)) {
            std::cout << "only support 1 k dim." << std::endl;
            return EinSumPassStatus::NOT_SUPPORT;
        }
        // 没有free轴时增加一根size为1的轴
        it = label_info.find(EinsumDimensionType::FREE);
        bool add_free_label = (it == label_info.end());

        // 多根free轴判断
        it = label_info.find(EinsumDimensionType::FREE);
        int32_t free_dim_size = 1;
        if (it != label_info.end()) {
            auto free_labels = it->second.labels;
            auto free_indices = it->second.indices;
            for (size_t j = 0; j < free_labels.size(); ++j) {
                auto dim_index = free_indices[j];
                auto dim_size = input_shape.GetDim(dim_index);
                if (free_dim_size > 1 && dim_size > 1) {
                    std::cout << "Multiple free axes that are not equal to 1 are not supported." << std::endl;
                    return EinSumPassStatus::NOT_SUPPORT;
                }
                if (dim_size > free_dim_size) {
                    free_dim_size = dim_size;
                }
            }
            for (size_t j = 0; j < free_labels.size(); ++j) {
                auto dim_index = free_indices[j];
                auto dim_size = input_shape.GetDim(dim_index);
                if ((free_dim_size != 1 && dim_size == 1) || (free_dim_size == 1 && i != 0)) {
                    remove_label[dim_index] = true;
                }
            }
        }

        // reduce轴判断
        it = label_info.find(EinsumDimensionType::REDUCE);
        if (it != label_info.end()) {
            auto reduce_labels = it->second.labels;
            auto reduce_indices = it->second.indices;
            for (size_t j = 0; j < reduce_labels.size(); ++j) {
                auto dim_index = reduce_indices[j];
                auto dim_size = input_shape.GetDim(dim_index);
                if (dim_size != 1) {
                    std::cout << "don't support reduce dim." << std::endl;
                    return EinSumPassStatus::NOT_SUPPORT;
                }
                remove_label[dim_index] = true;
            }
        }

        // 生成新的labels信息
        for (size_t j = 0; j < ori_label.size(); ++j) {
            // 删除reshape去掉的label
            if (!remove_label[j]) {
                new_label.push_back(ori_label[j]);
            }
        }
        if (add_free_label) {
            uint8_t new_free_label;
            if (GetOneNewSubscript(label_used, new_free_label) != EinSumPassStatus::SUCCESS) {
                std::cout << "no more subcript available" << std::endl;
                return EinSumPassStatus::ERROR;
            };
            new_label.push_back(new_free_label);
            dims_type_map[new_free_label] = EinsumDimensionType::FREE;
        }
    }
    return EinSumPassStatus::SUCCESS;
}

void LabelIndexMap(std::vector<std::vector<uint8_t>>& op_labels, std::vector<std::map<uint8_t, int>>& label_index_maps)
{
    // 获取原始的label和index映射关系
    for (size_t i = 0; i < op_labels.size(); ++i) {
        auto op_label = op_labels[i];
        for (size_t j = 0; j < op_label.size(); ++j) {
            auto label = op_label[j];
            label_index_maps[i][label] = static_cast<int>(j);
        }
    }
}

graphStatus CreateReshapeNode(GraphPtr& graph, GNode& node_reshape, std::string& node_name, GNode& node_input,
                              int32_t input_out_index, std::vector<int32_t>& reshape_size)
{
    TensorDesc input_desc;
    if (node_input.GetOutputDesc(input_out_index, input_desc) != GRAPH_SUCCESS) {
        std::cout << "create reshape node: get input desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    Operator op_reshape = op::Reshape(node_name.c_str());
    // 创建Shape, const节点
    std::string const_node_name = node_name + "/" + "shape";
    Shape const_shape = Shape({reshape_size.size()});
    TensorDesc const_shape_desc(const_shape, FORMAT_ND, DT_INT32);
    const_shape_desc.SetOriginShape(const_shape);
    Tensor shape_tensor(const_shape_desc, reinterpret_cast<const uint8_t*>(reshape_size.data()),
                        reshape_size.size() * sizeof(int32_t));
    auto const_shape_op = op::Const(const_node_name.c_str()).set_attr_value(shape_tensor);
    op_reshape.SetInput(kInputShape, const_shape_op);
    auto const_shape_node = graph->AddNodeByOp(const_shape_op);
    node_reshape = graph->AddNodeByOp(op_reshape);
    if (const_shape_node.UpdateOutputDesc(0, const_shape_desc) != GRAPH_SUCCESS) {
        std::cout << "create reshape node: update const shape output desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (node_reshape.UpdateInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        std::cout << "create reshape node: update reshape input desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (node_reshape.UpdateInputDesc(1, const_shape_desc) != GRAPH_SUCCESS) {
        std::cout << "create reshape node: update reshape shape desc failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(node_input, input_out_index, node_reshape, 0) != GRAPH_SUCCESS) {
        std::cout << "create reshape node: add data edge failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(const_shape_node, 0, node_reshape, 1) != GRAPH_SUCCESS) {
        std::cout << "create reshape node: add data edge failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (op_reshape.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "create reshape node: infer shape and type failed." << std::endl;
        return GRAPH_FAILED;
    }

    op_reshape.BreakConnect();
    return GRAPH_SUCCESS;
}

graphStatus CreateTransposeNode(GraphPtr& graph, GNode& node_transpose, std::string& node_name, GNode& node_input,
                                int32_t input_out_index, std::vector<int32_t>& perm_index)
{
    TensorDesc input_desc;
    if (node_input.GetOutputDesc(input_out_index, input_desc) != GRAPH_SUCCESS) {
        std::cout << "create transpose node: get output desc failed." << std::endl;
        return GRAPH_FAILED;
    }

    Operator op_transpose = op::Transpose(node_name.c_str());
    node_transpose = graph->AddNodeByOp(op_transpose);

    std::string const_perm_node_name = node_name + "/" + "perm";
    Shape const_perm_shape = Shape({perm_index.size()});
    TensorDesc const_perm_desc(const_perm_shape, FORMAT_ND, DT_INT32);
    const_perm_desc.SetOriginShape(const_perm_shape);
    Tensor perm_tensor(const_perm_desc, reinterpret_cast<const uint8_t*>(perm_index.data()),
                       perm_index.size() * sizeof(int32_t));
    auto const_perm_op = op::Const(const_perm_node_name.c_str()).set_attr_value(perm_tensor);
    op_transpose.SetInput(kInputPerm, const_perm_op);
    auto const_perm_node = graph->AddNodeByOp(const_perm_op);
    if (const_perm_node.UpdateOutputDesc(0, const_perm_desc) != GRAPH_SUCCESS) {
        std::cout << "create transpose node: update const perm output desc failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (node_transpose.UpdateInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        std::cout << "create transpose node: update transpose input desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (node_transpose.UpdateInputDesc(1, const_perm_desc) != GRAPH_SUCCESS) {
        std::cout << "create transpose node: update transpose perm desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(node_input, input_out_index, node_transpose, 0) != GRAPH_SUCCESS) {
        std::cout << "create transpose node: add data edge failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(const_perm_node, 0, node_transpose, 1) != GRAPH_SUCCESS) {
        std::cout << "create transpose node: add data edge failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (op_transpose.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "create transpose node: infer shape and type failed." << std::endl;
        return GRAPH_FAILED;
    }
    op_transpose.BreakConnect();
    return GRAPH_SUCCESS;
}

std::tuple<bool, bool, bool, bool> GetBmmParams(
    bool& swap_input, std::vector<std::vector<uint8_t>>& bmm_input_labels,
    std::map<uint8_t, EinsumDimensionType>& dims_type_map, std::vector<TensorDesc>& bmm_input_descs,
    std::vector<std::map<uint8_t, int>>& input_label_indexs, std::vector<uint8_t>& out_labels,
    std::vector<uint8_t>& batch_out_labels, std::vector<int32_t>& bmm_out_size, std::vector<int32_t>& out_reshape_size,
    std::vector<int32_t>& out_perm_index)
{
    // 获取batchmatmul需要的参数，1.左右输入是否转置，2.BatchMatmul结束后的label列表，3.reshape列表，4.transpose列表
    bool out_reshape = false;
    bool out_transpose = false;
    bool transpose_x1 = false;
    bool transpose_x2 = false;
    int32_t x1_idx = 0;
    int32_t x2_idx = 1;
    if (swap_input) {
        x1_idx = 1;
        x2_idx = 0;
    }
    TensorDesc x1_desc = bmm_input_descs[x1_idx];
    TensorDesc x2_desc = bmm_input_descs[x2_idx];
    // bmm后的shape，B轴、M轴、N轴
    // 根据out_label中上述label的前后顺序，确定transpose out_perm_index
    // 根据trans后的label列表，对比最终的label列表，确定reshape增轴位置
    std::vector<uint8_t> bmm_out_labels;

    for (size_t i = 0; i < batch_out_labels.size(); ++i) {
        auto label = batch_out_labels[i];
        auto label_it = find(bmm_input_labels[0].begin(), bmm_input_labels[0].end(), label);
        if (label_it != bmm_input_labels[0].end()) {
            auto dim_index = distance(bmm_input_labels[0].begin(), label_it);
            auto dim_size = x1_desc.GetShape().GetDim(dim_index);
            bmm_out_labels.push_back(label);
            bmm_out_size.push_back(static_cast<int32_t>(dim_size));
        }
    }

    int32_t x1_k_index = 0;
    int32_t x1_m_index = 0;
    for (size_t i = 0; i < bmm_input_labels[x1_idx].size(); ++i) {
        auto label = bmm_input_labels[x1_idx][i];
        auto dim_type = dims_type_map.find(label);
        if (dim_type != dims_type_map.end()) {
            if (dim_type->second == EinsumDimensionType::FREE) {
                auto dim_size = x1_desc.GetShape().GetDim(i);
                bmm_out_labels.push_back(label);
                bmm_out_size.push_back(static_cast<int32_t>(dim_size));
                x1_m_index = i;
            } else if (dim_type->second == EinsumDimensionType::SUM_K) {
                x1_k_index = i;
            }
        }
    }

    int32_t x2_k_index = 0;
    int32_t x2_n_index = 0;

    for (size_t i = 0; i < bmm_input_labels[x2_idx].size(); ++i) {
        auto label = bmm_input_labels[x2_idx][i];
        auto dim_type = dims_type_map.find(label);
        if (dim_type != dims_type_map.end()) {
            if (dim_type->second == EinsumDimensionType::FREE) {
                auto dim_size = x2_desc.GetShape().GetDim(i);
                bmm_out_labels.push_back(label);
                bmm_out_size.push_back(static_cast<int32_t>(dim_size));
                x2_n_index = i;
            } else if (dim_type->second == EinsumDimensionType::SUM_K) {
                x2_k_index = i;
            }
        }
    }
    //  判断transpose和reshape
    //  遍历equation中的输出labels
    //  1.搜索是否在bmm输出labels中，不在的话对应位置增轴，size为1
    //  2.如果在输出labels中，判断当前位置和原始位置是否相等，不等的话transpose为true

    int32_t transpose_dim_index = 0;

    for (size_t i = 0; i < out_labels.size(); ++i) {
        auto out_label = out_labels[i];
        auto bmm_out_find = find(bmm_out_labels.begin(), bmm_out_labels.end(), out_label);
        if (bmm_out_find != bmm_out_labels.end()) {
            auto bmm_out_dim_index = distance(bmm_out_labels.begin(), bmm_out_find);
            if (transpose_dim_index++ != bmm_out_dim_index) {
                out_transpose = true;
            }
            out_perm_index.push_back(static_cast<int32_t>(bmm_out_dim_index));
            out_reshape_size.push_back(static_cast<int32_t>(bmm_out_size[bmm_out_dim_index]));
        } else {
            out_reshape = true;
            out_reshape_size.push_back(static_cast<int32_t>(1));
        }
    }
    for (size_t i = 0; i < bmm_out_labels.size(); ++i) {
        auto out_label = bmm_out_labels[i];
        auto it = find(out_labels.begin(), out_labels.end(), out_label);
        if (it == out_labels.end()) {
            out_reshape = true;
            out_perm_index.insert(out_perm_index.begin() + i, static_cast<int32_t>(i));
        }
    }

    transpose_x1 = (x1_k_index < x1_m_index);
    transpose_x2 = (x2_k_index > x2_n_index);

    return std::make_tuple(out_transpose, out_reshape, transpose_x1, transpose_x2);
}

graphStatus BmmInput(GraphPtr& graph, GNode& node_bmm, bool& swap_input, std::vector<GNode>& bmm_input_nodes,
                     std::vector<int32_t>& input_indexs, std::vector<int32_t>& bmm_out_size, GNode& node_einsum,
                     std::string& einsum_node_name, bool& transpose_x1, bool& transpose_x2, bool& batch_flag)
{
    int32_t x1_idx = 0;
    int32_t x2_idx = 1;
    if (swap_input) {
        x1_idx = 1;
        x2_idx = 0;
    }
    // 获取原始einsum节点的输出描述信息
    TensorDesc einsum_out_desc;
    if (node_einsum.GetOutputDesc(0, einsum_out_desc) != GRAPH_SUCCESS) {
        std::cout << "get einsum node out desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    TensorDesc bmm_x1_desc;
    TensorDesc bmm_x2_desc;
    if (bmm_input_nodes[x1_idx].GetOutputDesc(input_indexs[x1_idx], bmm_x1_desc) != GRAPH_SUCCESS) {
        std::cout << "get bmm_input node x1 desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (bmm_input_nodes[x2_idx].GetOutputDesc(input_indexs[x2_idx], bmm_x2_desc) != GRAPH_SUCCESS) {
        std::cout << "get bmm_input node x2 desc failed." << std::endl;
        return GRAPH_FAILED;
    }

    Operator op_matmul;

    if (batch_flag) {
        std::string node_name = einsum_node_name + "/" + std::string(kBatchMatmulV2);
        op_matmul = op::BatchMatMulV2(node_name.c_str());
        node_bmm = graph->AddNodeByOp(op_matmul);
        ge::AscendString attr_adj_x1(kAttrAdjX1);
        ge::AscendString attr_adj_x2(kAttrAdjX2);
        if (node_bmm.SetAttr(attr_adj_x1, transpose_x1) != GRAPH_SUCCESS) {
            std::cout << "set bmm node attr adj_x1 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (node_bmm.SetAttr(attr_adj_x2, transpose_x2) != GRAPH_SUCCESS) {
            std::cout << "set bmm node attr adj_x1 failed" << std::endl;
            return GRAPH_FAILED;
        }
    } else {
        std::string node_name = einsum_node_name + "/" + std::string(kMatmulV2);
        op_matmul = op::MatMulV2(node_name.c_str());
        node_bmm = graph->AddNodeByOp(op_matmul);
        ge::AscendString attr_trans_x1(kAttrTransposeX1);
        ge::AscendString attr_trans_x2(kAttrTransposeX2);
        if (node_bmm.SetAttr(attr_trans_x1, transpose_x1) != GRAPH_SUCCESS) {
            std::cout << "set mm node attr transpose_x1 failed." << std::endl;
            return GRAPH_FAILED;
        }
        if (node_bmm.SetAttr(attr_trans_x2, transpose_x2) != GRAPH_SUCCESS) {
            std::cout << "set mm node attr transpose_x2 failed." << std::endl;
            return GRAPH_FAILED;
        }
    }
    // 更新inputDesc和output_desc;
    if (node_bmm.UpdateInputDesc(0, bmm_x1_desc) != GRAPH_SUCCESS) {
        std::cout << "update bmm node input x1 desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (node_bmm.UpdateInputDesc(1, bmm_x2_desc) != GRAPH_SUCCESS) {
        std::cout << "update bmm node input x2 desc failed." << std::endl;
        return GRAPH_FAILED;
    }
    if (graph->AddDataEdge(bmm_input_nodes[x1_idx], input_indexs[x1_idx], node_bmm, 0) != GRAPH_SUCCESS) {
        std::cout << "add data edge between bmm and input x1 failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (graph->AddDataEdge(bmm_input_nodes[x2_idx], input_indexs[x2_idx], node_bmm, 1) != GRAPH_SUCCESS) {
        std::cout << "add data edge between bmm and input x2 failed." << std::endl;
        return GRAPH_FAILED;
    }

    if (op_matmul.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "bmm node infershape and type failed." << std::endl;
        return GRAPH_FAILED;
    }
    op_matmul.BreakConnect();
    return GRAPH_SUCCESS;
}

std::tuple<bool, bool> GetInputProcessParams(std::vector<uint8_t>& ori_labels,
                                             std::vector<uint8_t>& labels_after_reshape,
                                             std::vector<uint8_t>& labels_after_trans, TensorDesc& op_desc,
                                             std::vector<uint8_t>& out_batch_labels, std::vector<int32_t>& reshape_size,
                                             std::vector<int32_t>& perm_index)
{
    bool reshape = false;
    bool transpose = false;
    reshape = CompareTwoVector(ori_labels, labels_after_reshape);
    Shape input_shape = op_desc.GetShape();
    if (reshape) {
        // 填充reshape_size
        for (size_t i = 0; i < labels_after_reshape.size(); ++i) {
            auto label = labels_after_reshape[i];
            auto it = find(ori_labels.begin(), ori_labels.end(), label);
            if (it != ori_labels.end()) {
                auto dim_index = distance(ori_labels.begin(), it);
                auto dim_size = input_shape.GetDim(dim_index);
                reshape_size.push_back(static_cast<int32_t>(dim_size));
            } else {
                // 轴不在原始输入中，为新增轴，size为1
                reshape_size.push_back(static_cast<int32_t>(1));
            }
        }
    }

    int32_t dim = 0;
    auto labels_before_trans = reshape ? labels_after_reshape : ori_labels;

    // 处理transpose
    for (size_t i = 0; i < out_batch_labels.size(); ++i) {
        auto batch_label = out_batch_labels[i];
        auto it = find(labels_before_trans.begin(), labels_before_trans.end(), batch_label);
        if (it != labels_before_trans.end()) {
            auto dim_index = distance(labels_before_trans.begin(), it);
            perm_index.push_back(static_cast<int32_t>(dim_index));
            if (dim++ != dim_index) {
                transpose = true;
            }
            labels_after_trans.push_back(batch_label);
        }
    }

    for (size_t i = 0; i < labels_before_trans.size(); ++i) {
        auto label = labels_before_trans[i];
        auto it = find(out_batch_labels.begin(), out_batch_labels.end(), label);
        if (it == out_batch_labels.end()) {
            perm_index.push_back(static_cast<int32_t>(i));
            labels_after_trans.push_back(label);
            if (dim++ != i) {
                transpose = true;
            }
        }
    }

    return std::make_tuple(reshape, transpose);
}

bool GetTransposeDst(std::vector<uint8_t>& ori_labels, std::vector<uint8_t>& out_batch_labels,
                     std::vector<int32_t>& permutation_index, std::vector<uint8_t>& op_labels_after_trans)
{
    bool transpose = false;
    int32_t dim = 0;
    // batch轴前移
    permutation_index.resize(ori_labels.size());
    for (size_t i = 0; i < out_batch_labels.size(); ++i) {
        auto batch_label = out_batch_labels[i];
        auto batch_label_index = find(ori_labels.begin(), ori_labels.end(), batch_label);
        if (batch_label_index != ori_labels.end()) {
            permutation_index[dim] = distance(ori_labels.begin(), batch_label_index);
            if (dim != permutation_index[dim]) {
                transpose = true;
            }
            dim++;
            op_labels_after_trans.push_back(batch_label);
        }
    }
    // 处理剩余轴
    for (size_t i = 0; i < ori_labels.size(); ++i) {
        auto label = ori_labels[i];
        auto batch_label_find = find(out_batch_labels.begin(), out_batch_labels.end(), label);
        if (batch_label_find == out_batch_labels.end()) {
            permutation_index[dim] = i;
            op_labels_after_trans.push_back(label);
            if (dim != i) {
                transpose = true;
            }
            dim++;
        }
    }
    return transpose;
}

graphStatus RelinkOutput(GraphPtr& graph, GNode& src_node, GNode& new_node, int32_t src_node_out_index)
{
    // 重连数据边
    std::vector<std::pair<GNodePtr, int32_t>> output_nodes = src_node.GetOutDataNodesAndPortIndexs(src_node_out_index);
    auto ret = GRAPH_SUCCESS;
    for (size_t j = 0; j < output_nodes.size(); j++) {
        ret = graph->RemoveEdge(src_node, src_node_out_index, *output_nodes[j].first, output_nodes[j].second);
        if (ret != GRAPH_SUCCESS) {
            return ret;
        }
        ret = graph->AddDataEdge(new_node, src_node_out_index, *output_nodes[j].first, output_nodes[j].second);
    }
    return ret;
}

EinSumPassStatus EinSumSplitProcess(GraphPtr& graph, CustomPassContext& custom_context, GNode& einsum_node,
                                    int32_t& affect_count)
{
    ge::AscendString name;
    einsum_node.GetName(name);
    std::string einsum_node_name(name.GetString());
    // 获取节点输入、输出、equation信息
    ge::AscendString attr_name_equation(kAttrEquation);
    ge::AscendString attr_value_equation;
    if (einsum_node.GetAttr(attr_name_equation, attr_value_equation) != GRAPH_SUCCESS) {
        std::cout << "Get attr equation failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }

    std::string equation = attr_value_equation.GetString();
    size_t num_ops = einsum_node.GetInputsSize();
    size_t output_num = einsum_node.GetOutputsSize();

    // 切分equation
    std::vector<std::vector<uint8_t>> op_labels(num_ops);
    std::vector<std::vector<uint8_t>> out_label_list(output_num);
    std::vector<uint8_t> label_used(TOTAL_LABELS, 0);

    EinSumPassStatus ret = SplitEquation(num_ops, output_num, equation, op_labels, out_label_list, label_used);
    if (ret != EinSumPassStatus::SUCCESS) {
        std::cout << "process split equation failed." << std::endl;
        return ret;
    }

    if (op_labels.size() != 2 || out_label_list.size() != 1) {
        std::cout << "only support 2 input and 1 output." << std::endl;
        return EinSumPassStatus::NOT_SUPPORT;
    }

    // 省略号替换处理
    std::vector<GNode> bmm_input_nodes;
    std::vector<int32_t> ori_input_out_indexs;
    std::vector<uint8_t> out_labels = out_label_list[0];
    for (size_t i = 0; i < op_labels.size(); ++i) {
        auto [node_input, out_index] = einsum_node.GetInDataNodesAndPortIndexs(i);
        bmm_input_nodes.push_back(*node_input);
        ori_input_out_indexs.push_back(out_index);
    }

    GNode bmm_output_node;
    bool swap_input = false;
    TensorDesc op_desc_0;
    TensorDesc op_desc_1;
    if (einsum_node.GetInputDesc(0, op_desc_0) != GRAPH_SUCCESS) {
        std::cout << "get einsum_node input_desc 0 failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }

    if (einsum_node.GetInputDesc(1, op_desc_1) != GRAPH_SUCCESS) {
        std::cout << "get einsum_node input_desc 1 failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }
    std::vector<TensorDesc> op_descs;
    op_descs.push_back(op_desc_0);
    op_descs.push_back(op_desc_1);

    ret = EllipsisReplace(op_descs, op_labels, out_labels, label_used);
    if (ret != EinSumPassStatus::SUCCESS) {
        std::cout << "ellipsis replace failed." << std::endl;
        return ret;
    }

    // 获取label和index原始映射关系
    std::vector<std::map<uint8_t, int>> input_label_index;
    input_label_index.resize(op_labels.size());
    std::map<uint8_t, int> out_label_index;
    LabelIndexMap(op_labels, input_label_index);

    // 检测是否有重复label，有的话表示取对角线操作，暂不支持，跳过
    for (size_t i = 0; i < op_labels.size(); ++i) {
        std::vector<uint8_t> labels = op_labels[i];
        std::set<uint8_t> labels_set(labels.begin(), labels.end());
        if (labels_set.size() != labels.size()) {
            std::cout << "Duplicate labels are not supported." << std::endl;
            return EinSumPassStatus::NOT_SUPPORT;
        }
    }
    std::set<uint8_t> out_labels_set(out_labels.begin(), out_labels.end());
    if (out_labels_set.size() != out_labels.size()) {
        std::cout << "output can't have Duplicate labels." << std::endl;
        return EinSumPassStatus::ERROR;
    }
    // 统计label出现次数
    std::vector<uint8_t> label_count(TOTAL_LABELS, 0);
    LabelCountMap(op_labels, label_count);
    // 判断label类型
    std::map<uint8_t, EinsumDimensionType> dims_type_map;
    GetDimsType(label_count, out_labels, dims_type_map);

    // batch校验拆分
    ret = BatchRevise(dims_type_map, op_labels, op_descs, label_used);
    if (ret != EinSumPassStatus::SUCCESS) {
        std::cout << "batchRevise failed." << std::endl;
        return ret;
    }

    // 收集每个input的label类型
    std::vector<std::map<EinsumDimensionType, LabelInfo>> input_label_infos;
    input_label_infos.resize(op_labels.size());
    std::vector<int64_t> reshape_i0_size;
    std::vector<int64_t> reshape_i1_size;
    std::vector<int64_t> reshape_o_size;
    std::vector<int64_t> transpose_i0_index;
    std::vector<int64_t> transpose_i1_index;
    std::vector<int64_t> transpose_i2_index;
    std::vector<uint8_t> batch_output_labels;
    // 输入的batch轴顺序和输出保持一致，transpose
    bool find_free = false;
    for (auto& label : out_labels) {
        if (dims_type_map[label] == EinsumDimensionType::BATCH) {
            batch_output_labels.push_back(label);
        } else if (dims_type_map[label] == EinsumDimensionType::FREE) {
            if (!find_free) {
                // 找下这个轴是否来自右输入
                auto dim_find = find(op_labels[1].begin(), op_labels[1].end(), label);
                if (dim_find != op_labels[1].end()) {
                    swap_input = true;
                }
                find_free = true;
            }
        }
    }

    std::vector<std::vector<uint8_t>> op_labels_after_reshape;
    op_labels_after_reshape.resize(op_labels.size());
    std::vector<std::vector<uint8_t>> op_labels_after_trans;
    op_labels_after_trans.resize(op_labels.size());
    std::vector<bool> unlink_input_flag(op_labels.size(), false);
    std::vector<TensorDesc> bmm_input_descs;
    // 校验两个输入的dim支不支持
    CollectDimensionType(op_labels, dims_type_map, input_label_infos);
    ret = CheckDimType(op_labels, bmm_input_nodes, ori_input_out_indexs, input_label_infos, dims_type_map,
                       op_labels_after_reshape, label_used);
    if (ret != EinSumPassStatus::SUCCESS) {
        std::cout << "dim type check failed, not support." << std::endl;
        return ret;
    }
    //  遍历输入，执行reshape和transpose
    for (size_t idx = 0; idx < op_labels.size(); ++idx) {
        std::vector<int32_t> input_reshape_size;
        std::vector<int32_t> input_perm_index;
        // 判断是否执行reshape
        auto [reshape, transpose] =
            GetInputProcessParams(op_labels[idx], op_labels_after_reshape[idx], op_labels_after_trans[idx],
                                  op_descs[idx], batch_output_labels, input_reshape_size, input_perm_index);
        if (graph->RemoveEdge(bmm_input_nodes[idx], ori_input_out_indexs[idx], einsum_node, idx) != GRAPH_SUCCESS) {
            std::cout << "remove input edge failed." << std::endl;
            return EinSumPassStatus::ERROR;
        }
        if (reshape) {
            GNode node_reshape;
            std::string reshape_node_name = einsum_node_name + "/input_" + to_string(idx) + "/" + std::string(kReshape);
            if (CreateReshapeNode(graph, node_reshape, reshape_node_name, bmm_input_nodes[idx],
                                  ori_input_out_indexs[idx], input_reshape_size) != GRAPH_SUCCESS) {
                std::cout << "create input reshape node failed." << std::endl;
                return EinSumPassStatus::ERROR;
            }
            bmm_input_nodes[idx] = node_reshape;
            ori_input_out_indexs[idx] = 0;
            op_labels[idx] = op_labels_after_reshape[idx];
        }
        if (transpose) {
            // 执行transpose
            GNode node_transpose;
            std::string transpose_node_name =
                einsum_node_name + "/input_" + to_string(idx) + "/" + std::string(kTranspose);
            if (CreateTransposeNode(graph, node_transpose, transpose_node_name, bmm_input_nodes[idx],
                                    ori_input_out_indexs[idx], input_perm_index) != GRAPH_SUCCESS) {
                std::cout << "create input transpose node failed." << std::endl;
                return EinSumPassStatus::ERROR;
            }
            op_labels[idx] = op_labels_after_trans[idx];
            bmm_input_nodes[idx] = node_transpose;
            ori_input_out_indexs[idx] = 0;
        }
        TensorDesc bmm_input_desc;
        if (bmm_input_nodes[idx].GetOutputDesc(ori_input_out_indexs[idx], bmm_input_desc) != GRAPH_SUCCESS) {
            std::cout << "get bmm_input_desc failed." << std::endl;
            return EinSumPassStatus::ERROR;
        }
        bmm_input_descs.push_back(bmm_input_desc);
    }

    bool bmm_flag = batch_output_labels.size() != 0;
    // 生成matmul相关参数
    std::vector<int32_t> bmm_out_size;
    std::vector<int32_t> out_reshape_size;
    std::vector<int32_t> out_perm_index;
    GNode node_bmm;

    auto [out_trans, out_reshape, transpose_x1, transpose_x2] =
        GetBmmParams(swap_input, op_labels, dims_type_map, bmm_input_descs, input_label_index, out_labels,
                     batch_output_labels, bmm_out_size, out_reshape_size, out_perm_index);

    if (BmmInput(graph, node_bmm, swap_input, bmm_input_nodes, ori_input_out_indexs, bmm_out_size, einsum_node,
                 einsum_node_name, transpose_x1, transpose_x2, bmm_flag) != GRAPH_SUCCESS) {
        std::cout << "create bmm node failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }

    bmm_output_node = node_bmm;

    if (out_trans) {
        GNode node_trans_out;
        std::string node_name = einsum_node_name + "/output" + "/" + std::string(kTranspose);
        if (CreateTransposeNode(graph, node_trans_out, node_name, bmm_output_node, 0, out_perm_index) !=
            GRAPH_SUCCESS) {
            std::cout << "create output transpose node failed." << std::endl;
            return EinSumPassStatus::ERROR;
        }
        bmm_output_node = node_trans_out;
    }
    if (out_reshape) {
        GNode node_reshape_out;
        std::string node_name = einsum_node_name + "/output" + "/" + std::string(kReshape);
        if (CreateReshapeNode(graph, node_reshape_out, node_name, bmm_output_node, 0, out_reshape_size) !=
            GRAPH_SUCCESS) {
            std::cout << "create output transpose node failed." << std::endl;
            return EinSumPassStatus::ERROR;
        }
        bmm_output_node = node_reshape_out;
    }

    if (RelinkOutput(graph, einsum_node, bmm_output_node, 0) != GRAPH_SUCCESS) {
        std::cout << "relink output edge failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }
    if (graph->RemoveNode(einsum_node) != GRAPH_SUCCESS) {
        std::cout << "remove einsum node failed." << std::endl;
        return EinSumPassStatus::ERROR;
    }

    affect_count++;
    return EinSumPassStatus::SUCCESS;
}

graphStatus EinSumSplitPass(GraphPtr& graph, CustomPassContext& custom_context)
{
    std::cout << "----------start on EinsumPass----------" << std::endl;
    int32_t affect_count = 0;
    // 找到EinSum节点
    std::vector<GNode> nodes = graph->GetAllNodes();
    for (GNode& node : nodes) {
        if (CheckNodeType(node, kEinSum)) {
            auto einsum_ret = EinSumSplitProcess(graph, custom_context, node, affect_count);
            if (einsum_ret == EinSumPassStatus::ERROR) {
                return GRAPH_FAILED;
            }
        }
    }
    std::cout << "end on EinsumPass, affect_count is: " << affect_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("EinSumSplitPass").CustomPassFn(EinSumSplitPass).Stage(CustomPassStage::kAfterInferShape);
