# !/usr/bin/env python3
# -- coding: utf-8 --
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import dataclasses
from typing import Dict, DefaultDict, List, Tuple, Set

from tensorflow import Operation, Tensor
from tensorflow.core.framework.graph_pb2 import GraphDef


# DefaultDict:
#     Key: Tensor => Represent output tensor of `IteratorGetNext` operation.
#     Val: List[Tuple[int, Operation]] => Contains target operation of output tensor and it's corresponding index.
ReplacementSpec = DefaultDict[Tensor, List[Tuple[int, Operation]]]


@dataclasses.dataclass
class AnchorRecord:
    replacement_spec: ReplacementSpec
    passing_tensors: List[Tensor]
    batch_tensor_indexs: List[int]
    sub_cutting_points: List[Tensor]
    sub_graph_def: GraphDef
    input_names: List[str]
    output_names: List[str]
    is_training: bool
    input_indexs: List[int] = None


@dataclasses.dataclass
class SubgraphInfo:
    subgraph_in: Dict[Operation, Set[Operation]]
    subgraph_out: Dict[Operation, Set[Operation]]
    subgraph_to_push: Set[Operation]
