#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterator
from dataclasses import dataclass
import torch_npu
import torch
import numpy as np
from torch.utils.data.dataset import IterableDataset
from torchrec.streamable import Pipelineable
from torchrec import KeyedJaggedTensor, JaggedTensor


@dataclass
class Batch(Pipelineable):
    sparse_features: KeyedJaggedTensor
    labels: torch.Tensor

    def __init__(self, sparse_features, labels) -> None:
        self.sparse_features = sparse_features
        self.labels = labels

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch_npu.npu.streams.Stream) -> None:
        self.sparse_features.record_stream(stream)
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
        )


class RandomRecDataset(IterableDataset[Batch]):
    def __init__(self, batch_num, lookup_lens, num_embeddings, table_num):
        super().__init__()
        self.index = 0
        self.lookup_lens = lookup_lens
        self.num_embeddings = num_embeddings
        self.table_num = table_num
        self.batch_num = batch_num
        torch.manual_seed(1)
        self.data = [self.generate_one_batch() for _ in range(batch_num)]

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        feature_len = len(self.num_embeddings)
        for ind in reversed(range(feature_len)):
            name = f"feat{ind}"
            id_range = self.num_embeddings[ind]
            ids = torch.randint(0, id_range, (self.lookup_lens,))
            lengths = torch.ones(self.lookup_lens).long()
            input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label)


class RandomRecDatasetUnique(RandomRecDataset):

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        feature_len = len(self.num_embeddings)
        for ind in reversed(range(feature_len)):
            name = f"feat{ind}"
            id_range = self.num_embeddings[ind]
            choice_ids = np.random.choice(id_range, (self.lookup_lens,), replace=False)
            ids = torch.Tensor(choice_ids).long()
            lengths = torch.ones(self.lookup_lens).long()
            input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label)
