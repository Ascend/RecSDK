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

from typing import Dict, Any, List

from rec_sdk_common.log.log import LoggingProxy as logger
from mx_rec.core.emb.dynamic_sparse_embedding import DynamicSparseEmbedding


class MergeableSparseEmbedding(DynamicSparseEmbedding):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._merged_small_tables: List[str] = []

    def __str__(self) -> str:
        safe_attr_keys = ("_table_name", "_merged_small_tables")
        attrs = [f"'{k}': {v}" for k, v in self.__dict__.items() if k in safe_attr_keys]
        attr_fmt_str = ", ".join(attrs)
        fmt_str = "{} => <{}>".format(self.__class__, attr_fmt_str)

        return fmt_str

    def __repr__(self) -> str:
        attrs = [f"'{k}': {v}" for k, v in self.__dict__.items()]
        attr_fmt_repr = ", ".join(attrs)
        fmt_repr = "{} => <{}>".format(self.__class__, attr_fmt_repr)

        return fmt_repr

    @property
    def merged_small_tables(self) -> List[str]:
        return self._merged_small_tables

    def merge_in(self, small_table_name: str, config: Dict[str, Any]) -> None:
        if small_table_name in self._merged_small_tables:
            raise ValueError(
                "given table => '{}' already exists in mergeable table => '{}'".format(
                    small_table_name, self.table_name
                )
            )

        self._merged_small_tables.append(small_table_name)

        # Merge padding keys.
        if self._padding_keys_mask:
            self._padding_keys.extend(config.get("padding_keys"))
            self._padding_keys = list(set(self._padding_keys))  # Remove duplicates.
            self._padding_keys_len += config.get("padding_keys_len")

    def _set_slice_vocab_size(self):
        """Device vocabulary size will be forced set to 1 in dynamic expansion mode."""

        self._slice_device_vocabulary_size = 1
        logger.info(
            "'%s''s sliced HBM vacabulary size has been set to '1' due to dynamic expansion mode.",
            self.table_name,
        )
