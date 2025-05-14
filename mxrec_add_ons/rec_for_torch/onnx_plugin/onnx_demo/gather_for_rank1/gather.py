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