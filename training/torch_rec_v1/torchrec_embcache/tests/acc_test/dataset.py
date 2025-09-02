#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch_npu
from torch.utils.data.dataset import IterableDataset
from torchrec_embcache.sparse.jagged_tensor_with_timestamp import (
    JaggedTensorWithTimestamp,
    KeyedJaggedTensorWithTimestamp
)

from torchrec import KeyedJaggedTensor, JaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class Batch(Pipelineable):
    sparse_features: KeyedJaggedTensor
    labels: torch.Tensor

    def __init__(self, sparse_features, labels) -> None:
        self.sparse_features = sparse_features
        self.labels = labels

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features,
            labels=self.labels.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch_npu.npu.streams.Stream) -> None:
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
        )


class RandomRecDataset(IterableDataset[Batch]):
    def __init__(self, batch_num, lookup_lens, num_embeddings, table_num,
                 is_evict_enabled: bool = False, timestamp_min: int = None, timestamp_max: int = None):
        super().__init__()
        self.index = 0
        self.lookup_lens = lookup_lens
        self.num_embeddings = num_embeddings
        self.table_num = table_num
        self.batch_num = batch_num
        torch.manual_seed(1)

        # 淘汰相关参数
        self.is_evict_enabled = is_evict_enabled
        timestamp_is_not_none = timestamp_min is not None or timestamp_max is not None
        timestamp_illegal = False if not timestamp_is_not_none else timestamp_min >= timestamp_max
        if is_evict_enabled and timestamp_illegal:
            raise ValueError("The timestamp param invalid, timestamp_min is greater than or equal to timestamp_max,"
                             f" timestamp_min:{timestamp_min}, timestamp_max:{timestamp_max}.")
        default_start_time = int(time.mktime(time.struct_time((2023, 5, 5, 14, 33, 20, 0, 0, 0))))  # 1683268400
        default_end_time = int(time.mktime(time.struct_time((2025, 5, 5, 14, 33, 20, 0, 0, 0))))  # 1746426800
        # evict_threshold范围：[now() - end_time, now() - start_time]
        self.start_time = timestamp_min or default_start_time
        self.end_time = timestamp_max or default_end_time

        self.data = [self.generate_one_batch() for _ in range(batch_num)]

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        feature_len = len(self.num_embeddings)
        if self.is_evict_enabled:
            for ind in range(feature_len):
                name = f"feat{ind}"
                id_range = self.num_embeddings[ind]
                ids = torch.randint(0, id_range, (self.lookup_lens,))
                timestamp_data = torch.randint(self.start_time, self.end_time, ids.size(), dtype=torch.int64)
                lengths = torch.ones(self.lookup_lens).long()
                input_dict[name] = JaggedTensorWithTimestamp(values=ids, lengths=lengths, timestamps=timestamp_data)
            kjt_tensor = KeyedJaggedTensorWithTimestamp.from_jt_dict(input_dict)
        else:
            for ind in range(feature_len):
                name = f"feat{ind}"
                id_range = self.num_embeddings[ind]
                ids = torch.randint(0, id_range, (self.lookup_lens,))
                lengths = torch.ones(self.lookup_lens).long()
                input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
            kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)

        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label)
