#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Dict, List

import torch

from hybrid_torchrec.modules.little_embedding import (
    EmbeddingConfig,
)
from torchrec import JaggedTensor, KeyedJaggedTensor


class Ids2IndexSample:
    def __init__(self):
        self.max_index: int = 0
        self.ids2index_map: Dict[int, int] = defaultdict(int)

    def compute_index(self, jt: JaggedTensor) -> List[int]:
        ids_list: torch.Tensor = jt.values()
        hash_indices: List[int] = []
        for ids in ids_list:
            ids = ids.item()
            if ids in self.ids2index_map:
                hash_indices.append(self.ids2index_map[ids])
            else:
                index = self.max_index
                self.ids2index_map[ids] = index
                hash_indices.append(index)
                self.max_index += 1
        return hash_indices


class Ids2IndexSampleHandler:
    def __init__(self, configs: List[EmbeddingConfig]):
        self.table_hash_samples: Dict[str, Ids2IndexSample] = defaultdict(Ids2IndexSample)
        for config in configs:
            self.table_hash_samples[config.table_name] = Ids2IndexSample()

    def compute_index(self, kjt: KeyedJaggedTensor) -> Dict[str, List[int]]:
        jt_dict = kjt.to_dict()
        hash_index_dict: Dict[str, List[int]] = {}
        for feat_name in jt_dict.keys():
            table_hash_sample: Ids2IndexSample = self.table_hash_samples[feat_name]
            hash_index_dict[feat_name] = table_hash_sample.compute_index(jt_dict[feat_name])
        return hash_index_dict
