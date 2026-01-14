/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstring>
#include <algorithm>
#include <iostream>
#include "register_custom_pass.h"
#include "all_ops.h"

using namespace std;
using namespace ge;

std::vector<uint8_t> convert_int64_to_uint8(const std::vector<int64_t>& input)
{
    std::vector<uint8_t> output;
    output.reserve(input.size() * 4);

    for (int64_t elem : input) {
        int32_t val_32 = static_cast<int32_t>(elem);

        uint8_t bytes[4];
        std::memcpy(bytes, &val_32, sizeof(val_32));

        output.push_back(bytes[0]);
        output.push_back(bytes[1]);
        output.push_back(bytes[2]);
        output.push_back(bytes[3]);
    }

    return output;
}

graphStatus BroadcastEqualPass(GraphPtr& graph, CustomPassContext& custom_context)
{
    std::cout << "----------start broadcastEqualPass----------" << std::endl;
    int32_t affect_count = 0;
    std::vector<GNode> nodes = graph->GetAllNodes();
    graphStatus ret = GRAPH_FAILED;

    for (GNode& node : nodes) {
        ge::AscendString type;
        node.GetType(type);
        std::string node_type(type.GetString());
        ge::AscendString name;
        node.GetName(name);
        std::string node_name(name.GetString());

        if (node_type != "Equal") {
            continue;
        }

        GNode equal_node = node;

        TensorDesc equal_input_desc_0;
        ret = equal_node.GetInputDesc(0, equal_input_desc_0);
        if (ret != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        TensorDesc equal_input_desc_1;
        ret = equal_node.GetInputDesc(1, equal_input_desc_1);
        if (ret != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        if (equal_input_desc_0.GetDataType() != ge::DT_INT64 || equal_input_desc_1.GetDataType() != ge::DT_INT64) {
            continue;
        }

        auto out_nodes = equal_node.GetOutDataNodesAndPortIndexs(0);
        if (out_nodes.size() != 1) {
            continue;
        }

        auto reduce_node = out_nodes[0].first;
        ge::AscendString reduce_type;
        reduce_node->GetType(reduce_type);

        if (reduce_type != "ReduceAny") {
            continue;
        }

        TensorDesc reduce_axis_desc;
        ret = reduce_node->GetInputDesc(1, reduce_axis_desc);
        if (ret != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }

        DataType reduce_axis_dtype = reduce_axis_desc.GetDataType();
        if (reduce_axis_dtype != DT_INT32) {
            continue;
        }

        auto reduce_axis_node = reduce_node->GetInDataNodesAndPortIndexs(1).first;
        ge::AscendString reduce_axis_type;
        reduce_axis_node->GetType(reduce_axis_type);

        if (reduce_axis_type != "Const" && reduce_axis_type != "Constant") {
            continue;
        }

        Tensor reduce_axis_tensor;
        reduce_node->GetInputConstData(1, reduce_axis_tensor);
        uint8_t* data = reduce_axis_tensor.GetData();
        int32_t axis;
        std::memcpy(&axis, data, sizeof(int32_t));

        Shape equal_input_desc_0_shape = equal_input_desc_0.GetShape();
        std::vector<int64_t> equal_input_desc_0_dims = equal_input_desc_0_shape.GetDims();
        Shape equal_input_desc_1_shape = equal_input_desc_1.GetShape();
        std::vector<int64_t> equal_input_desc_1_dims = equal_input_desc_1_shape.GetDims();
        if (equal_input_desc_0_dims.size() != equal_input_desc_1_dims.size()) {
            continue;
        }

        if (axis < 0) {
            axis = equal_input_desc_0_dims.size() + axis;
        }

        if (axis >= equal_input_desc_0_dims.size() || axis < 0) {
            return GRAPH_FAILED;
        }

        TensorDesc equal_output_desc;
        ret = equal_node.GetOutputDesc(0, equal_output_desc);
        if (ret != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        Shape equal_output_desc_shape = equal_output_desc.GetShape();
        std::vector<int64_t> equal_output_desc_dims = equal_output_desc_shape.GetDims();

        std::vector<int64_t> equal_input_desc_to_update, equal_input_desc_to_update_peer;
        std::vector<int64_t> equal_ret_reshape_output_dims;
        equal_ret_reshape_output_dims = equal_output_desc_dims;
        if (equal_input_desc_0_dims[axis] == 1 && equal_input_desc_1_dims[axis] != 1) {
            equal_input_desc_to_update = equal_input_desc_1_dims;
            equal_input_desc_to_update_peer = equal_input_desc_0_dims;
        } else if (equal_input_desc_0_dims[axis] != 1 && equal_input_desc_1_dims[axis] == 1) {
            equal_input_desc_to_update = equal_input_desc_0_dims;
            equal_input_desc_to_update_peer = equal_input_desc_1_dims;
        } else {
            continue;
        }

        int32_t target_axis = -1;
        for (int32_t i = axis - 1; i >= 0; i--) {
            if (equal_input_desc_to_update[i] != 1 && equal_input_desc_to_update_peer[i] == 1) {
                target_axis = i + 1;
                break;
            }
            if (equal_input_desc_to_update[i] != 1 && equal_input_desc_to_update_peer[i] != 1) {
                break;
            }
        }

        if (target_axis == -1) {
            for (size_t i = axis + 1; i < equal_input_desc_to_update.size(); i++) {
                if (equal_input_desc_to_update[i] != 1 && equal_input_desc_to_update_peer[i] == 1) {
                    target_axis = i - 1;
                    break;
                }
                if (equal_input_desc_to_update[i] != 1 && equal_input_desc_to_update_peer[i] != 1) {
                    break;
                }
            }
        }

        if (target_axis == axis || target_axis == -1) {
            continue;
        }

        if (target_axis < axis) {
            std::rotate(equal_input_desc_0_dims.begin() + target_axis, equal_input_desc_0_dims.begin() + axis,
                        equal_input_desc_0_dims.begin() + axis + 1);
            std::rotate(equal_input_desc_1_dims.begin() + target_axis, equal_input_desc_1_dims.begin() + axis,
                        equal_input_desc_1_dims.begin() + axis + 1);
            std::rotate(equal_input_desc_to_update.begin() + target_axis, equal_input_desc_to_update.begin() + axis,
                        equal_input_desc_to_update.begin() + axis + 1);
            std::rotate(equal_ret_reshape_output_dims.begin() + target_axis,
                        equal_ret_reshape_output_dims.begin() + axis, equal_ret_reshape_output_dims.begin() + axis + 1);
        } else {
            std::rotate(equal_input_desc_0_dims.begin() + axis, equal_input_desc_0_dims.begin() + axis + 1,
                        equal_input_desc_0_dims.begin() + target_axis);
            std::rotate(equal_input_desc_1_dims.begin() + axis, equal_input_desc_1_dims.begin() + axis + 1,
                        equal_input_desc_1_dims.begin() + target_axis);
            std::rotate(equal_input_desc_to_update.begin() + axis, equal_input_desc_to_update.begin() + axis + 1,
                        equal_input_desc_to_update.begin() + target_axis);
            std::rotate(equal_ret_reshape_output_dims.begin() + axis, equal_ret_reshape_output_dims.begin() + axis + 1,
                        equal_ret_reshape_output_dims.begin() + target_axis);
        }

        std::vector<int64_t> reduce_output_dims;
        reduce_output_dims = equal_ret_reshape_output_dims;
        reduce_output_dims.erase(reduce_output_dims.begin() + target_axis);

        std::vector<int64_t> reshape_0_output_dims, reshape_1_output_dims, equal_output_dims;
        int64_t reshape_0_output_dim, reshape_1_output_dim, equal_output_dim;
        int64_t previous_dim;
        for (size_t i = 0; i < equal_input_desc_0_dims.size(); i++) {
            if (i == 0) {
                previous_dim = equal_input_desc_0_dims[0];
                reshape_0_output_dim = equal_input_desc_0_dims[0];
                reshape_1_output_dim = equal_input_desc_1_dims[0];
                equal_output_dim = equal_ret_reshape_output_dims[0];
                continue;
            }
            if ((previous_dim == 1 && equal_input_desc_0_dims[i] == 1) ||
                (previous_dim > 1 && equal_input_desc_0_dims[i] > 1 && equal_input_desc_1_dims[i] == 1)) {
                reshape_0_output_dim *= equal_input_desc_0_dims[i];
                reshape_1_output_dim *= equal_input_desc_1_dims[i];
                equal_output_dim *= equal_ret_reshape_output_dims[i];
            } else {
                reshape_0_output_dims.emplace_back(reshape_0_output_dim);
                reshape_1_output_dims.emplace_back(reshape_1_output_dim);
                equal_output_dims.emplace_back(equal_output_dim);
                reshape_0_output_dim = equal_input_desc_0_dims[i];
                reshape_1_output_dim = equal_input_desc_1_dims[i];
                equal_output_dim = equal_ret_reshape_output_dims[i];
                if (equal_input_desc_0_dims[i] > 1 && equal_input_desc_1_dims[i] > 1) {
                    previous_dim = -1;
                } else {
                    previous_dim = equal_input_desc_0_dims[i];
                }
            }
        }
        reshape_0_output_dims.emplace_back(reshape_0_output_dim);
        reshape_1_output_dims.emplace_back(reshape_1_output_dim);
        equal_output_dims.emplace_back(equal_output_dim);

        auto equal_input0_node = equal_node.GetInDataNodesAndPortIndexs(0);
        auto equal_input1_node = equal_node.GetInDataNodesAndPortIndexs(1);
        auto reduce_output_nodes = reduce_node->GetOutDataNodesAndPortIndexs(0);

        std::string reshape_0_name = std::string(name.GetString()) + "/InputReshape0";
        std::string reshape_1_name = std::string(name.GetString()) + "/InputReshape1";
        std::string equal_output_name = std::string(name.GetString()) + "/OutputReshape";

        std::string reshape_0_axis_name = std::string(name.GetString()) + "/InputReshape0/Axis";
        std::string reshape_1_axis_name = std::string(name.GetString()) + "/InputReshape1/Axis";
        std::string equal_output_axis_name = std::string(name.GetString()) + "/OutputReshape/Axis";
        std::string new_reduce_axis_name = std::string(name.GetString()) + "/NewReduceAxis";

        TensorDesc equal_input_0_reshape_input_desc(equal_input_desc_0);
        Shape equal_input_0_reshape_output_shape(reshape_0_output_dims);
        TensorDesc equal_input_0_reshape_output_desc(equal_input_0_reshape_output_shape, FORMAT_ND, DT_INT64);
        equal_input_0_reshape_output_desc.SetOriginShape(equal_input_0_reshape_output_shape);

        TensorDesc equal_input_1_reshape_input_desc(equal_input_desc_1);
        Shape equal_input_1_reshape_output_shape(reshape_1_output_dims);
        TensorDesc equal_input_1_reshape_output_desc(equal_input_1_reshape_output_shape, FORMAT_ND, DT_INT64);
        equal_input_1_reshape_output_desc.SetOriginShape(equal_input_1_reshape_output_shape);

        Shape reduce_output_shape(reduce_output_dims);
        TensorDesc reduce_output_desc(reduce_output_shape, FORMAT_ND, DT_BOOL);
        reduce_output_desc.SetOriginShape(reduce_output_shape);

        Shape reduce_new_input_shape(equal_ret_reshape_output_dims);
        TensorDesc reduce_new_input_desc(reduce_new_input_shape, FORMAT_ND, DT_BOOL);
        TensorDesc reduce_new_output_desc(reduce_output_desc);
        reduce_new_input_desc.SetOriginShape(reduce_new_input_shape);

        Shape equal_output_reshape_input_shape(equal_output_dims);
        TensorDesc equal_output_reshape_input_desc(equal_output_reshape_input_shape, FORMAT_ND, DT_BOOL);
        TensorDesc equal_output_reshape_output_desc(reduce_new_input_desc);
        equal_output_reshape_input_desc.SetOriginShape(equal_output_reshape_input_shape);

        TensorDesc equal_new_input_0_desc(equal_input_0_reshape_output_desc);
        TensorDesc equal_new_input_1_desc(equal_input_1_reshape_output_desc);
        TensorDesc equal_new_output_desc(equal_output_reshape_input_desc);

        std::vector<int64_t> reshape_const_before_combined_shape_list, reshape_const_after_combined_shape_list;
        reshape_const_before_combined_shape_list.emplace_back(equal_ret_reshape_output_dims.size());
        reshape_const_after_combined_shape_list.emplace_back(reshape_0_output_dims.size());
        Shape reshape_const_before_combined(reshape_const_before_combined_shape_list);
        Shape reshape_const_after_combined(reshape_const_after_combined_shape_list);
        Shape new_reduce_axis_shape({1});

        TensorDesc reduce_0_axis_desc(reshape_const_after_combined, FORMAT_ND, DT_INT32);
        reduce_0_axis_desc.SetOriginShape(reshape_const_after_combined);
        TensorDesc reduce_0_axis_const_desc(reduce_0_axis_desc);
        TensorDesc reduce_1_axis_desc(reshape_const_after_combined, FORMAT_ND, DT_INT32);
        reduce_1_axis_desc.SetOriginShape(reshape_const_after_combined);
        TensorDesc reduce_1_axis_const_desc(reduce_1_axis_desc);
        TensorDesc equal_output_reshape_axis_desc(reshape_const_before_combined, FORMAT_ND, DT_INT32);
        equal_output_reshape_axis_desc.SetOriginShape(reshape_const_before_combined);
        TensorDesc equal_output_reshape_axis_const_desc(equal_output_reshape_axis_desc);
        TensorDesc new_reduce_axis_desc(new_reduce_axis_shape, FORMAT_ND, DT_INT32);
        new_reduce_axis_desc.SetOriginShape(new_reduce_axis_shape);

        op::Reshape reshape_0_op(reshape_0_name.c_str());
        GNode reshape_0_gnode = graph->AddNodeByOp(reshape_0_op);
        if (reshape_0_gnode.UpdateInputDesc(0, equal_input_0_reshape_input_desc) != GRAPH_SUCCESS) {
            std::cout << "update reshape 0 input desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reshape_0_gnode.UpdateInputDesc(1, reduce_0_axis_desc) != GRAPH_SUCCESS) {
            std::cout << "update reshape 0 axis desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reshape_0_gnode.UpdateOutputDesc(0, equal_input_0_reshape_output_desc) != GRAPH_SUCCESS) {
            std::cout << "update reshape 0 output desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        op::Const reshape_0_axis(reshape_0_axis_name.c_str());
        TensorDesc reshape_0_axis_op_desc(reduce_0_axis_desc);
        reshape_0_axis_op_desc.SetOriginShape(reshape_const_after_combined);
        std::vector<uint8_t> equal_input_desc_0_dims_data = convert_int64_to_uint8(equal_input_desc_0_dims);
        Tensor reshape_0_axis_tensor(reshape_0_axis_op_desc, equal_input_desc_0_dims_data);
        reshape_0_axis.set_attr_value(reshape_0_axis_tensor);
        GNode reshape_0_axis_gnode = graph->AddNodeByOp(reshape_0_axis);
        if (reshape_0_axis_gnode.UpdateOutputDesc(0, reduce_0_axis_const_desc) != GRAPH_SUCCESS) {
            std::cout << "update reduce 0 axis const desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        op::Reshape reshape_1_op(reshape_1_name.c_str());
        GNode reshape_1_gnode = graph->AddNodeByOp(reshape_1_op);
        if (reshape_1_gnode.UpdateInputDesc(0, equal_input_1_reshape_input_desc) != GRAPH_SUCCESS) {
            std::cout << "update reshape 1 input desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reshape_1_gnode.UpdateInputDesc(1, reduce_1_axis_desc) != GRAPH_SUCCESS) {
            std::cout << "update reshape 1 axis desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reshape_1_gnode.UpdateOutputDesc(0, equal_input_1_reshape_output_desc) != GRAPH_SUCCESS) {
            std::cout << "update reshape 1 output desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        op::Const reshape_1_axis(reshape_1_axis_name.c_str());
        TensorDesc reshape_1_axis_op_desc(reduce_1_axis_desc);
        reshape_1_axis_op_desc.SetOriginShape(reshape_const_after_combined);
        std::vector<uint8_t> equal_input_desc_1_dims_data = convert_int64_to_uint8(equal_input_desc_1_dims);
        Tensor reshape_1_axis_tensor(reshape_1_axis_op_desc, equal_input_desc_1_dims_data);
        reshape_1_axis.set_attr_value(reshape_1_axis_tensor);
        GNode reshape_1_axis_gnode = graph->AddNodeByOp(reshape_1_axis);
        if (reshape_1_axis_gnode.UpdateOutputDesc(0, reduce_1_axis_const_desc) != GRAPH_SUCCESS) {
            std::cout << "update reduce 1 axis const desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        op::Reshape equal_output_reshape_op(equal_output_name.c_str());
        GNode equal_output_reshape_gnode = graph->AddNodeByOp(equal_output_reshape_op);
        if (equal_output_reshape_gnode.UpdateInputDesc(0, equal_output_reshape_input_desc) != GRAPH_SUCCESS) {
            std::cout << "update equal output reshape input desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (equal_output_reshape_gnode.UpdateInputDesc(1, equal_output_reshape_axis_desc) != GRAPH_SUCCESS) {
            std::cout << "update equal output reshape axis desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (equal_output_reshape_gnode.UpdateOutputDesc(0, equal_output_reshape_output_desc) != GRAPH_SUCCESS) {
            std::cout << "update equal output reshape output desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        op::Const equal_output_reshape_axis(equal_output_axis_name.c_str());
        TensorDesc equal_output_reshape_axis_op_desc(equal_output_reshape_axis_desc);
        equal_output_reshape_axis_op_desc.SetOriginShape(reshape_const_before_combined);
        std::vector<uint8_t> equal_output_dims_data = convert_int64_to_uint8(reduce_output_dims);
        Tensor equal_output_reshape_axis_tensor(equal_output_reshape_axis_op_desc, equal_output_dims_data);
        equal_output_reshape_axis.set_attr_value(equal_output_reshape_axis_tensor);
        GNode equal_output_reshape_axis_gnode = graph->AddNodeByOp(equal_output_reshape_axis);
        if (equal_output_reshape_axis_gnode.UpdateOutputDesc(0, equal_output_reshape_axis_const_desc) !=
            GRAPH_SUCCESS) {
            std::cout << "update equal output reshape axis const desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        op::Const new_reduce_axis(new_reduce_axis_name);
        TensorDesc new_reduce_axis_op_desc(new_reduce_axis_desc);
        new_reduce_axis_op_desc.SetOriginShape(new_reduce_axis_shape);
        Tensor new_reduce_axis_tensor(new_reduce_axis_op_desc, reinterpret_cast<uint8_t*>(&target_axis), 4);
        new_reduce_axis.set_attr_value(new_reduce_axis_tensor);
        GNode new_reduce_axis_gnode = graph->AddNodeByOp(new_reduce_axis);
        if (new_reduce_axis_gnode.UpdateOutputDesc(0, new_reduce_axis_desc) != GRAPH_SUCCESS) {
            std::cout << "update new reduce axis axis const desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (equal_node.UpdateInputDesc(0, equal_new_input_0_desc) != GRAPH_SUCCESS) {
            std::cout << "update new equal input 0 desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (equal_node.UpdateInputDesc(1, equal_new_input_1_desc) != GRAPH_SUCCESS) {
            std::cout << "update new equal input 1 desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (equal_node.UpdateOutputDesc(0, equal_new_output_desc) != GRAPH_SUCCESS) {
            std::cout << "update new equal output desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reduce_node->UpdateInputDesc(0, reduce_new_input_desc) != GRAPH_SUCCESS) {
            std::cout << "update new reduce input desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reduce_node->UpdateInputDesc(1, new_reduce_axis_desc) != GRAPH_SUCCESS) {
            std::cout << "update new reduce axis desc failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (reduce_node->UpdateOutputDesc(0, reduce_new_output_desc) != GRAPH_SUCCESS) {
            std::cout << "update new reduce output desc failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->RemoveEdge(*equal_input0_node.first, equal_input0_node.second, equal_node, 0) != GRAPH_SUCCESS) {
            std::cout << "remove edge equal input 0 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->RemoveEdge(*equal_input1_node.first, equal_input1_node.second, equal_node, 1) != GRAPH_SUCCESS) {
            std::cout << "remove edge equal input 1 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->RemoveEdge(equal_node, 0, *reduce_node, 0) != GRAPH_SUCCESS) {
            std::cout << "remove edge equal to reduce failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->RemoveNode(*reduce_axis_node) != GRAPH_SUCCESS) {
            std::cout << "remove reduce axis node failed" << std::endl;
            return GRAPH_FAILED;
        }
        for (auto out_node : reduce_output_nodes) {
            if (graph->RemoveEdge(*reduce_node, 0, *out_node.first, out_node.second) != GRAPH_SUCCESS) {
                std::cout << "remove reduce to output edge failed" << std::endl;
                return GRAPH_FAILED;
            }
        }

        if (graph->AddDataEdge(*equal_input0_node.first, equal_input0_node.second, reshape_0_gnode, 0) !=
            GRAPH_SUCCESS) {
            std::cout << "add edge equal input 0 to reshape 0 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->AddDataEdge(reshape_0_axis_gnode, 0, reshape_0_gnode, 1) != GRAPH_SUCCESS) {
            std::cout << "add reshape axis 0 to reshape 0 failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(reshape_0_gnode, 0, equal_node, 0) != GRAPH_SUCCESS) {
            std::cout << "add reshape 0 to equal 0 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->AddDataEdge(*equal_input1_node.first, equal_input1_node.second, reshape_1_gnode, 0) !=
            GRAPH_SUCCESS) {
            std::cout << "add edge equal input 1 to reshape 1 failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(reshape_1_axis_gnode, 0, reshape_1_gnode, 1) != GRAPH_SUCCESS) {
            std::cout << "add reshape axis 1 to reshape 1 failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(reshape_1_gnode, 0, equal_node, 1) != GRAPH_SUCCESS) {
            std::cout << "add reshape 1 to equal 1 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->AddDataEdge(equal_node, 0, equal_output_reshape_gnode, 0) != GRAPH_SUCCESS) {
            std::cout << "add equal node 0 to equal output reshape 0 failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->AddDataEdge(equal_output_reshape_axis_gnode, 0, equal_output_reshape_gnode, 1) != GRAPH_SUCCESS) {
            std::cout << "add equal equal output reshape axis to equal output reshape failed" << std::endl;
            return GRAPH_FAILED;
        }

        if (graph->AddDataEdge(equal_output_reshape_gnode, 0, *reduce_node, 0) != GRAPH_SUCCESS) {
            std::cout << "add equal equal output reshape to reduce failed" << std::endl;
            return GRAPH_FAILED;
        }
        if (graph->AddDataEdge(new_reduce_axis_gnode, 0, *reduce_node, 1) != GRAPH_SUCCESS) {
            std::cout << "add new reduce axis to reduce failed" << std::endl;
            return GRAPH_FAILED;
        }
        for (auto out_node : reduce_output_nodes) {
            if (graph->AddDataEdge(*reduce_node, 0, *out_node.first, out_node.second) != GRAPH_SUCCESS) {
                std::cout << "add reduce ret reshapeto output edge failed" << std::endl;
                return GRAPH_FAILED;
            }
        }
        affect_count++;
    }

    std::cout << "end broadcastEqualPass, affect count:" << affect_count << std::endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("BroadcastEqualPass").CustomPassFn(BroadcastEqualPass).Stage(CustomPassStage::kAfterInferShape);
