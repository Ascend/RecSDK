#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from onnx import TensorProto
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node, make_graph
from onnx.checker import check_model
import onnx

BATCH_SIZE = 1
MAX_SEQ_LEN = 768
NUM_HEAD = 4
ATTENTION_DIM = 32
SILU_SCALE = 1 / MAX_SEQ_LEN
MASK_TYPE = 3
LAYOUT = "normal"

qkv_shape = [BATCH_SIZE, MAX_SEQ_LEN, NUM_HEAD, ATTENTION_DIM]

q = make_tensor_value_info("q", TensorProto.FLOAT16, qkv_shape)
k = make_tensor_value_info("k", TensorProto.FLOAT16, qkv_shape)
v = make_tensor_value_info("v", TensorProto.FLOAT16, qkv_shape)
timestamp_bias = make_tensor_value_info("timestamp_bias", TensorProto.FLOAT16, [BATCH_SIZE, MAX_SEQ_LEN, MAX_SEQ_LEN])
position_bias = make_tensor_value_info("position_bias", TensorProto.FLOAT16, [1, MAX_SEQ_LEN, MAX_SEQ_LEN])
mask = make_tensor_value_info("mask", TensorProto.FLOAT16, [BATCH_SIZE, 1, MAX_SEQ_LEN, MAX_SEQ_LEN])

attn_output = make_tensor_value_info("attn_output", TensorProto.FLOAT16,
    [BATCH_SIZE, MAX_SEQ_LEN, 3 * NUM_HEAD * ATTENTION_DIM])

node = make_node("HstuDenseForwardFuxi", ["q", "k", "v", "timestamp_bias", "position_bias", "mask"], ["attn_output"])
node.attribute.append(make_attribute("SILU_SCALE", SILU_SCALE))
node.attribute.append(make_attribute("maxSeqLen", MAX_SEQ_LEN))
node.attribute.append(make_attribute("MASK_TYPE", MASK_TYPE))
node.attribute.append(make_attribute("LAYOUT", LAYOUT))

graph = make_graph([node], "hstu_fuxi", [q, k, v, timestamp_bias, position_bias, mask], [attn_output])

model_def = make_model(graph, producer_name="hstu_fuxi-onnx")

model_def.opset_import[0].version = 11

onnx.save(model_def, "hstu_fuxi.onnx")