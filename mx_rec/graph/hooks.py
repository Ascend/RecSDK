#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from typing import List

import tensorflow as tf
from tensorflow import Operation, Graph

from mx_rec.util.log import logger
from mx_rec.graph.slicers import LookupSubgraphSlicer, OrphanLookupKeySlicer
from mx_rec.validator.validator import ClassValidator, para_checker_decorator


@para_checker_decorator(
    check_option_list=[
        ("op_types", ClassValidator, {"classes": (list)}),
        ("full_graph", ClassValidator, {"classes": (Graph, type(None))}),
    ]
)
class LookupSubgraphSlicerHook(tf.estimator.SessionRunHook):
    def __init__(self, op_types: List[Operation], full_graph: Graph = None) -> None:
        super().__init__()
        self._op_types = op_types
        self._full_graph = full_graph

    def begin(self) -> None:
        slicer = LookupSubgraphSlicer(self._op_types, self._full_graph)

        logger.info("Starts to summarize sliceable specific operations in lookup subgraph!")
        slicer.summarize()

        logger.info("Starts to slice specific operations and their corresponding minimum dependency graphs!")
        slicer.slice()


@para_checker_decorator(check_option_list=[("full_graph", ClassValidator, {"classes": (Graph, type(None))})])
class OrphanLookupKeySlicerHook(tf.estimator.SessionRunHook):
    def __init__(self, full_graph: Graph = None) -> None:
        super().__init__()
        self._full_graph = full_graph

    def begin(self) -> None:
        slicer = OrphanLookupKeySlicer(self._full_graph)

        logger.info("Starts to summarize sliceable orphan lookup keys!")
        slicer.summarize()

        logger.info("Starts to slice orphan lookup keys and their corresponding minimum dependency graphs!")
        slicer.slice()
