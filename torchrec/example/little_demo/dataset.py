# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import itertools
from typing import Iterator
from dataclasses import dataclass
import torch_npu
from torch.utils.data.dataset import IterableDataset
import torch
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
    def __init__(self, batch_size, batch_num, feat_names, id_ranges):
        super().__init__()
        self.index = 0
        self.names = list(itertools.chain.from_iterable(feat_names))
        self.id_ranges = id_ranges
        self.data = [self.generate_one_batch(batch_size) for i in range(batch_num)]

    def generate_one_batch(self, batch_size) -> Batch:
        torch.manual_seed(1)
        input_dict = {}
        for name, id_range in zip(self.names, self.id_ranges):
            ids = torch.randint(0, id_range, (batch_size,))
            lengths = torch.ones(batch_size).long()
            input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (batch_size,))
        return Batch(kjt_tensor, label)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
