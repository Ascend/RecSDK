#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import abc
from typing import Generic, Optional, Tuple, TypeVar
from dataclasses import dataclass
import numpy as np
import torch


TableOptionType = TypeVar("TableOptionType")
OptimizerInterface = TypeVar("OptimizerInterface")

torch_dtype_to_np_dtype = {
    torch.uint64: np.uint64,
    torch.int64: np.int64,
    torch.float32: np.float32,
}

KEY_TYPE = torch.int64
EMBEDDING_TYPE = torch.float32
SCORE_TYPE = torch.int64
OPT_STATE_TYPE = torch.float32

DEFAULT_LOAD_BATCH_SIZE = 65536


@dataclass
class StorageOptionalFilePaths:
    score_file_path: Optional[str] = None
    opt_file_path: Optional[str] = None


# make it standalone to avoid recursive references.
class Storage(abc.ABC, Generic[TableOptionType, OptimizerInterface]):
    @abc.abstractmethod
    def __init__(
        self,
        options: TableOptionType,
        optimizer: OptimizerInterface,
    ):
        pass

    @abc.abstractmethod
    def find(
        self,
        unique_keys: torch.Tensor,
        unique_vals: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_embeddings(
        self,
        unique_keys: torch.Tensor,
        unique_embs: torch.Tensor,
        founds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def insert(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def update(
        self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_missing: torch.Tensor
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def enable_update(self) -> bool:
        ...

    @abc.abstractmethod
    def dump(
        self,
        meta_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        optional_files: Optional[StorageOptionalFilePaths] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def load(
        self,
        meta_file_path: str,
        emb_file_path: str,
        embedding_file_path: str,
        include_optim: bool,
        optional_files: Optional[StorageOptionalFilePaths] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def embedding_dtype(
        self,
    ) -> torch.dtype:
        pass

    @abc.abstractmethod
    def embedding_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def value_dim(
        self,
    ) -> int:
        pass

    @abc.abstractmethod
    def init_optimizer_state(
        self,
    ) -> float:
        pass


class Cache(abc.ABC):
    @abc.abstractmethod
    def find(
            self,
            unique_keys: torch.Tensor,
            unique_vals: torch.Tensor,
            founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_embeddings(
            self,
            unique_keys: torch.Tensor,
            unique_embs: torch.Tensor,
            founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def find_missed_keys(
            self,
            unique_keys: torch.Tensor,
            founds: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def insert_and_evict(
            self,
            keys: torch.Tensor,
            values: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_evicted: int
        evicted_keys: torch.Tensor
        evicted_values: torch.Tensor
        evicted_scores: torch.Tensor
        return num_evicted, evicted_keys, evicted_values, evicted_scores

    @abc.abstractmethod
    def update(
            self, keys: torch.Tensor, grads: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        num_missing: int
        missing_keys: torch.Tensor
        missing_indices: torch.Tensor
        return num_missing, missing_keys, missing_indices

    @abc.abstractmethod
    def flush(self, storage: Storage) -> None:
        pass

    @abc.abstractmethod
    def reset(
            self,
    ) -> None:
        pass

    @abc.abstractmethod
    def cache_metrics(
            self,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def set_record_cache_metrics(self, record: bool) -> None:
        pass
