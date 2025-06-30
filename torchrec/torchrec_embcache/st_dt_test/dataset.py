#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
from dataclasses import dataclass
from typing import Iterator

import torch
import torch_npu
from torch.utils.data.dataset import IterableDataset

from torchrec import KeyedJaggedTensor, JaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class Batch(Pipelineable):
    sparse_features: KeyedJaggedTensor
    labels: torch.Tensor

    def __init__(self, sparse_features: KeyedJaggedTensor, labels: torch.Tensor, instances: int = 1) -> None:
        self.sparse_features = sparse_features
        for i in range(instances):
            setattr(self, f"instance{i}_sparse_features", sparse_features)
        self.labels = labels
        self.instances = instances

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features,
            labels=self.labels.to(device, non_blocking=non_blocking),
            instances=self.instances,
        )

    def record_stream(self, stream: torch_npu.npu.streams.Stream) -> None:
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
            instances=self.instances,
        )


class RandomRecDataset(IterableDataset[Batch]):
    def __init__(
            self, 
            batch_num, 
            lookup_lens, 
            num_embeddings, 
            table_num, 
            feature_names_lst=None, 
            generated_ids=None, 
            instances=1
        ):
        super().__init__()
        self.index = 0
        self.lookup_lens = lookup_lens
        self.num_embeddings = num_embeddings
        self.table_num = table_num
        self.batch_num = batch_num
        self.generated_ids = generated_ids
        self.feature_names_lst = feature_names_lst if feature_names_lst else [[f"feat{i}"] for i in range(table_num)]
        self.instances = instances
        torch.manual_seed(1)
        self.data = [self.generate_one_batch() for _ in range(batch_num)]

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        for ind, feature_names in enumerate(self.feature_names_lst):
            id_range = self.num_embeddings[ind]
            for feature_name in feature_names:
                ids = torch.randint(0, max(1, id_range), (self.lookup_lens,))
                lengths = torch.ones(self.lookup_lens).long()
                input_dict[feature_name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label, self.instances)


class FeatureNameNotInConfigRecDataset(RandomRecDataset):

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        for ind, feature_names in enumerate(self.feature_names_lst):
            feature_names = [f"error_{feature_name}" for feature_name in feature_names]
            id_range = self.num_embeddings[ind]
            for feature_name in feature_names:
                ids = torch.randint(0, max(1, id_range), (self.lookup_lens,))
                lengths = torch.ones(self.lookup_lens).long()
                input_dict[feature_name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label, self.instances)


class BoundOutOfRangeRecDataset(RandomRecDataset):

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        for ind, feature_names in enumerate(self.feature_names_lst):
            id_range = self.num_embeddings[ind]
            for i, feature_name in enumerate(feature_names):
                ids = []
                for _ in range(self.lookup_lens):
                    if self.generated_ids[ind][i]:
                        ids.append(self.generated_ids[ind][i].pop())
                    else:
                        ids.append(random.randint(0, id_range))
                ids = torch.tensor(ids)
                lengths = torch.ones(self.lookup_lens).long()
                input_dict[feature_name] = JaggedTensor(values=ids, lengths=lengths)
                self.generated_ids[ind].update(ids.tolist())
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label, self.instances)
