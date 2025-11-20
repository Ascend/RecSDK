#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from dataclasses import dataclass
from typing import Iterator, Optional, List, Union

import numpy as np
import torch
import torch_npu
from torch.utils.data.dataset import IterableDataset

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
    def __init__(self, batch_num, lookup_lens, num_embeddings, table_num, keys_per_table: Union[int, List[int]] = 1):
        super().__init__()
        self.index = 0
        self.lookup_lens = lookup_lens
        self.num_embeddings = num_embeddings
        self.table_num = table_num
        self.batch_num = batch_num
        self.keys_per_table = keys_per_table
        torch.manual_seed(1)
        self.data = [self.generate_one_batch() for _ in range(batch_num)]

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        feature_len = len(self.num_embeddings)
        if isinstance(self.keys_per_table, list) and len(self.keys_per_table) == len(self.num_embeddings):
            for ind in range(feature_len):
                for j in range(self.keys_per_table[ind]):
                    name = f"feat{ind}_key{j}"  # 此处name需要和创建EmbeddingConfig时对应
                    id_range = self.num_embeddings[ind]
                    ids = torch.randint(0, id_range, (self.lookup_lens,))
                    lengths = torch.ones(self.lookup_lens).long()
                    input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
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


class RandomRecDatasetV2(IterableDataset[Batch]):
    def __init__(self, batch_num, lookup_lens: int, num_embeddings, table_num, rank,
                 id_repeat_rate: Optional[float] = None, save_dataset: bool = False) -> None:
        super().__init__()
        self.index = 0
        self.lookup_lens = lookup_lens
        self.num_embeddings = num_embeddings
        self.table_num = table_num
        self.batch_num = batch_num
        self.rank = rank
        self.save_dataset = save_dataset
        if id_repeat_rate is not None and isinstance(id_repeat_rate, float):
            if not 0.0 <= id_repeat_rate <= 1.0:
                raise ValueError("param id_repeat_rate is invalid, is must be in range:[0.0, 1.0]")
        self.id_repeat_rate = id_repeat_rate
        torch.manual_seed(1)
        np.random.seed(42)
        self.data_file = "./dataset.pt"
        self.data = []
        if not os.path.exists(self.data_file):
            logging.info("generate dataset by random.")
            self.data = [self.generate_one_batch() for _ in range(batch_num)]
            if self.save_dataset and self.rank == 0:
                self._save_data_to_local_file()
        else:
            logging.info("generate dataset from saved file.")
            self._load_data_from_local_file()

    def _load_data_from_local_file(self):
        saved_data_list = torch.load(self.data_file, map_location="cpu")
        for save_data in saved_data_list:
            kjt = KeyedJaggedTensor(keys=save_data["keys"],
                                    values=save_data["values"],
                                    lengths=save_data["lengths"],
                                    offsets=save_data["offsets"])
            label = torch.tensor(save_data["labels"], dtype=torch.long)
            self.data.append(Batch(kjt, label))

    def _save_data_to_local_file(self):
        save_data_list = []
        for batch in self.data:
            kjt: KeyedJaggedTensor = batch.sparse_features
            save_data_list.append({
                "keys": kjt.keys(),
                "values": kjt.values(),
                "lengths": kjt.lengths(),
                "offsets": kjt.offsets(),
                "labels": batch.labels
            })
        torch.save(save_data_list, self.data_file)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def _generate_ids_by_repeat_rate(self, id_range, lookup_len) -> torch.Tensor:
        # generate unique id
        unique_ids_num = int(lookup_len * (1 - self.id_repeat_rate))
        unique_ids = np.random.choice(np.arange(0, id_range), size=unique_ids_num, replace=False)
        # generate repeat id
        repeat_ids_num = lookup_len - unique_ids_num
        repeat_ids = np.random.choice(unique_ids, size=repeat_ids_num, replace=True)
        # concat
        return torch.tensor(np.concatenate([unique_ids, repeat_ids], axis=0))

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        feature_len = len(self.num_embeddings)
        for ind in range(feature_len):
            self._full_jagged_tensor_dict(ind, input_dict)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label)

    def _full_jagged_tensor_dict(self, ind, input_dict):
        name = f"feat{ind}"
        id_range = self.num_embeddings[ind]
        if not self.id_repeat_rate:
            ids = torch.randint(0, id_range, (self.lookup_lens,))
        else:
            ids = self._generate_ids_by_repeat_rate(id_range, self.lookup_lens)
        lengths = torch.ones(self.lookup_lens).long()
        input_dict[name] = JaggedTensor(values=ids, lengths=lengths)