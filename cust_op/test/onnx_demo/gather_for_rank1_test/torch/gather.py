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

import torch
import torch_npu
from onnx import TensorProto
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node, make_graph
from onnx.checker import check_model
import onnx

x = make_tensor_value_info("x", TensorProto.FLOAT16, [2048])
index = make_tensor_value_info("index", TensorProto.INT32, [128])
y = make_tensor_value_info("y", TensorProto.FLOAT16, [128])

node = make_node("GatherForRank1", ["x", "index"], ["y"])

graph = make_graph([node], 'gather', [x, index], [y])

model_def = make_model(graph, producer_name='gather-onnx')

model_def.opset_import[0].version = 11

onnx.save(model_def, "gather.onnx")