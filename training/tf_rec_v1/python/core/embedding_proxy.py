#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

import collections
from typing import Any, Dict, List, Optional, Union

from rec_sdk_common.log.log import LoggingProxy as logger
from mx_rec.core.emb.mergeable_sparse_embedding import MergeableSparseEmbedding
from mx_rec.util.singleton import singleton

UnionKey = collections.namedtuple(
    typename="UnionKey",
    field_names=[
        "key_dtype",
        "emb_dim",
        "initializer_type",
        "is_save",
        "is_dp",
        "init_param",
        "all2all_gradients_op",
        "padding_keys_mask",
    ],
)


@singleton
class MergeableEmbeddingTableProxy:
    """A proxy class used for merging embedding table automatically.

    This class should be a singleton during once graph building process. In estimator mode,
    train and evalution will build graph seperately, In order to solve the problem of
    independent graph in evalution of estimator mode, this singleton should be reset after
    graph modification.
    """

    _MERGEABLE_TABLE_PREFIX = "mergeable_table"

    def __init__(self) -> None:
        self._mtable_id: int = 0
        self._mtables: List[MergeableSparseEmbedding] = []
        self._ukey_to_mtable: Dict[UnionKey, MergeableSparseEmbedding] = {}
        self._stable_to_mtable: Dict[str, MergeableSparseEmbedding] = {}

    @classmethod
    def _validate_small_tname(cls, tname: str) -> None:
        if tname.startswith(cls._MERGEABLE_TABLE_PREFIX):
            raise ValueError(
                "original table name => '{}' is not supposed to start with '{}'".format(
                    tname, cls._MERGEABLE_TABLE_PREFIX
                )
            )

    def reset(self) -> None:
        """Reset singleton after graph modification, because estimator will build graph in each evalution."""

        self.__init__()
        logger.info("'{}' singleton has been reset after graph was frozen".format(self.__class__))

    def create_mergeable_table(
        self, union_key: UnionKey, small_table_name: str, config: Dict[str, Any]
    ) -> MergeableSparseEmbedding:
        """Create a `MergeableSparseEmbedding` through this proxy class.

        This function will modify `table_name` in config dict, all mergeable table name during
        graph building process should be managed by this proxy class. Besides, no mergeable embedding
        table instance should be created directly, all meregable table instance should be created
        through proxy class.

        There are at least two mapping relation have to be maintained during graph building process:
            1. union key -> mergeable table instance.
            2. small table name -> mergeable table instance.

        Args:
            union_key: Immutable named tuple used for determine merge group of a new table.
            small_table_name: Original small table name.
            config: A config dict required by all sparse embedding classes.

        Raises:
            ValueError: if param 'small_table_name' startswith 'mergeable_table'.

        Returns:
            A new mergeable embedding table of class 'MergeableSparseEmbedding'.
        """

        self._validate_small_tname(small_table_name)

        # Due to inheritance restriction, `config` have to update its 'table_name'.
        config["table_name"] = self._gen_mergeable_table_name()

        mergeable_table = MergeableSparseEmbedding(config)
        mergeable_table.merge_in(small_table_name, config)

        self._mtables.append(mergeable_table)
        self._ukey_to_mtable[union_key] = mergeable_table
        self._stable_to_mtable[small_table_name] = mergeable_table

        logger.info(
            "A new mergeable embedding table '%s' has been created from embedding table '%s'.",
            mergeable_table.table_name,
            small_table_name,
        )

        return mergeable_table

    def find_mergeable_table(self, key: Union[UnionKey, str]) -> Optional[MergeableSparseEmbedding]:
        if isinstance(key, UnionKey):
            return self._ukey_to_mtable.get(key)
        elif isinstance(key, str):
            self._validate_small_tname(key)
            return self._stable_to_mtable.get(key)
        else:
            invalid_type = type(key)
            ukey_type = type(UnionKey)
            raise TypeError("not supported key type => '{}', expected '{}' or 'str'".format(invalid_type, ukey_type))

    def join_mergeable_table(
        self, mergeable_table: MergeableSparseEmbedding, small_table_name: str, config: Dict[str, Any]
    ) -> MergeableSparseEmbedding:
        if small_table_name in mergeable_table.merged_small_tables:
            raise ValueError(
                "given table name => '{}' has already joined mergeable table => '{}'.".format(
                    small_table_name, mergeable_table.table_name
                )
            )

        mergeable_table.merge_in(small_table_name, config)
        self._stable_to_mtable[small_table_name] = mergeable_table
        logger.info(
            "Mergeable embedding table '{}' has merged in a new small table '{}'.".format(
                mergeable_table.table_name, small_table_name
            )
        )

        return mergeable_table

    def _gen_mergeable_table_name(self) -> str:
        mergeable_tname = "{}_{}".format(self._MERGEABLE_TABLE_PREFIX, self._mtable_id)
        self._mtable_id += 1

        return mergeable_tname


def create_mergeable_embedding(small_table_name: str, config: Dict[str, Any], union_key: UnionKey):
    mtable_proxy = MergeableEmbeddingTableProxy()

    mergeable_table = mtable_proxy.find_mergeable_table(union_key)
    if mergeable_table:
        mtable_proxy.join_mergeable_table(mergeable_table, small_table_name, config)
    else:
        mergeable_table = mtable_proxy.create_mergeable_table(union_key, small_table_name, config)

    return mergeable_table
