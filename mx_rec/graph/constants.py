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


from enum import Enum


class DeprecatedOp(Enum):
    DEPRECATED_ITERATOR_GET_NEXT = "DEPRECATED_ITERATOR_GET_NEXT"
    DEPRECATED_PREFETCH_DATASET = "DEPRECATED_PREFETCH_DATASET"


class AnchorDatasetOp(Enum):
    MODEL_DATASET = "ModelDataset"
    OPTIMIZE_DATASET = "OptimizeDataset"
    PREFETCH_DATASET = "PrefetchDataset"


class AnchorIteratorOp(Enum):
    ITERATOR_GET_NEXT = "IteratorGetNext"
    ITERATOR_V2 = "IteratorV2"
    MAKE_ITERATOR = "MakeIterator"
    ONE_SHOT_ITERATOR = "OneShotIterator"
