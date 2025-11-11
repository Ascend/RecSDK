#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from typing import Any, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops

from rec_sdk_common.util.tf_adapter import hccl_ops, gen_npu_cpu_ops
from mxrec.python.constants.constants import MPLookupParams, LOCAL_EMBEDDING_COLLECTION
from mxrec.python.embedding.lookup.base_lookup import BaseLookup
from mxrec.python.embedding.table.static_emb_table import StaticEmbTable


class MPLookup(BaseLookup):
    """Embedding lookup for model parallel, and the hash strategy for the lookup key is modulo."""

    def __init__(self, emb_table: Union[StaticEmbTable], ids: tf.Tensor):
        super(MPLookup, self).__init__(emb_table, ids)

    def lookup(self) -> tf.Tensor:
        """Embedding lookup for model parallel.

        Under model parallelism, the embedding lookup is divided into the following steps:
            Step1: unique and relocate ids.
            Step2: get local embedding.
            Step3: get own embedding.
            Step4: get restore embedding.
            Step5: get sorted embedding.
            Step6: reshape to [bs, seq_len, emb_dim].

        Returns:
            The lookup result.
        """

        @tf.custom_gradient
        def _lookup_forward(embedding: tf.Tensor) -> Any:
            def _lookup_backward(embedding_grad: tf.Tensor) -> ops.IndexedSlices:
                embedding_grad = tf.reshape(embedding_grad, [-1, self._emb_table.dim], name="embedding_grad")

                if self._rank_size > 1:
                    restore_embedding_grad = tf.gather(
                        embedding_grad,
                        lookup_params.sorted_ids_indices,
                        name="restore_embedding_grad",
                    )
                else:
                    restore_embedding_grad = embedding_grad

                own_embedding_grad = tf.compat.v1.unsorted_segment_sum(
                    data=restore_embedding_grad,
                    segment_ids=lookup_params.local_ids_restore,
                    num_segments=tf.shape(own_embedding)[0],
                    name="own_embedding_grad",
                )

                local_embedding_grad = self._embedding_all2all(
                    own_embedding_grad,
                    lookup_params,
                    "local_embedding_grad",
                    is_bp=True,
                )

                grad = ops.IndexedSlices(
                    values=local_embedding_grad,
                    indices=lookup_params.local_ids,
                    dense_shape=tf.shape(embedding),
                )
                return grad

            # Step3: get own embedding.
            own_embedding = self._embedding_all2all(embedding, lookup_params, name="own_embedding")

            # Step4: get restore embedding.
            restore_embedding = tf.gather(own_embedding, lookup_params.local_ids_restore, name="restore_embedding")

            # Step5: get sorted embedding.
            if self._rank_size > 1:
                sorted_embedding = tf.compat.v1.scatter_nd(
                    lookup_params.sorted_ids_indices[:, tf.newaxis],
                    restore_embedding,
                    tf.shape(restore_embedding),
                    name="sorted_embedding",
                )
            else:
                sorted_embedding = restore_embedding

            # Step6: reshape to [bs, seq_len, emb_dim].
            res_shape = tf.concat(
                (
                    tf.shape(self._ids, out_type=self._emb_table.key_dtype),
                    (self._emb_table.dim,),
                ),
                axis=0,
            )
            lookup_res = tf.reshape(sorted_embedding, res_shape, name="lookup_res")

            return lookup_res, _lookup_backward

        with tf.compat.v1.variable_scope(self._get_default_lookup_name()):
            # Step1: unique and relocate ids.
            lookup_params = self._process_ids(self._ids)

            # Step2: get local embedding.
            table_handle = gen_npu_cpu_ops.table_to_resource_v2(table_id=[self._emb_table.table_id])
            local_embedding = gen_npu_cpu_ops.embedding_hash_table_lookup_or_insert(
                table_handle=table_handle,
                keys=lookup_params.local_ids,
                bucket_size=self._emb_table.slice_dev_vocab_size,
                embedding_dim=self._emb_table.dim,
            )
            tf.compat.v1.add_to_collection(LOCAL_EMBEDDING_COLLECTION, local_embedding)
            BaseLookup.set_local_emb_to_table_ins(local_embedding, self._emb_table)
            return _lookup_forward(local_embedding)

    def _process_ids(self, ids: tf.Tensor) -> MPLookupParams:
        """Deduplicate and relocate the feature ids.

        For example, in beginning, 2 ranks has there feature ids:
            rank0: 1, 2, 1, 3
            rank1: 2, 6, 1, 5
        After sort(Reorder the keys, the keys of rank 0 are placed in the front, and the keys of rank 1 are placed
        behind), each rank get:
            rank0: 2, 1, 1, 3
            rank1: 2, 6, 1, 5
        After local unique, each rank get:
            rank0: 2, 1, 3
            rank1: 2, 6, 1, 5
        After relocation, each rank get:
            rank0: 2, 2, 6
            rank1: 1, 3, 1, 5

        Args:
            ids: feature ids.

        Returns:
            A dataclass for the lookup parameters.

        """
        ids = tf.reshape(ids, shape=(-1,))

        if self._rank_size == 1:
            u_ids, u_idx, u_cnts = tf.unique_with_counts(ids)

            if self._emb_table.count_filter:
                u_ids = self._emb_table.count_filter.count_and_filter(keys=u_ids, cnts=u_cnts)

            if self._emb_table.time_evictor:
                u_ids = self._emb_table.time_evictor.update_last_timestamp(keys=u_ids)

            lookup_params = MPLookupParams(local_ids=u_ids, local_ids_restore=u_idx)
            return lookup_params

        # Reorder the ids. For example, if there are a total of 2 ranks, the ids of rank 0 are placed in the front,
        # and the ids of rank 1 are placed behind. This operation facilitates ALL2ALL to send and receive data.
        mask = tf.cast(tf.math.mod(ids, self._rank_size), tf.int32)
        sorted_indices = tf.argsort(mask)
        sorted_ids = tf.gather(ids, sorted_indices)

        local_ids: Optional[tf.Tensor] = None
        sc_all: Optional[tf.Tensor] = None
        u_idx: Optional[tf.Tensor] = None

        if not self._emb_table.count_filter:
            local_ids, sc_all, u_idx = self._get_local_ids(sorted_ids)
        else:
            local_ids, sc_all, u_idx = self._get_local_ids_with_filter(sorted_ids)

        if self._emb_table.time_evictor:
            local_ids = self._emb_table.time_evictor.update_last_timestamp(local_ids)

        # Record lookup parameters.
        sc_matrix = tf.reshape(sc_all, shape=(self._rank_size, self._rank_size))
        lookup_params = MPLookupParams(
            local_ids=local_ids,
            local_ids_restore=u_idx,
            sorted_ids_indices=sorted_indices,
            send_count_matrix=sc_matrix,
        )
        return lookup_params

    def _get_local_ids(self, sorted_ids: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Unique ids.
        u_ids, u_idx = tf.unique(x=sorted_ids)

        # Relocation ids.
        send_count = gen_math_ops.bincount(
            tf.cast(tf.math.mod(u_ids, self._rank_size), tf.int32),
            self._rank_size,
            tf.constant([], tf.int64),
        )
        sc_all = hccl_ops.allgather(send_count, self._rank_size)
        local_ids = hccl_ops.all_to_all_v_c(send_data=u_ids, send_count_matrix=sc_all, rank=self._rank_id)

        return local_ids, sc_all, u_idx

    def _get_local_ids_with_filter(self, sorted_ids: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Unique ids.
        u_ids, u_idx, u_cnts = tf.unique_with_counts(x=sorted_ids)

        # Relocation ids.
        send_count = gen_math_ops.bincount(
            tf.cast(tf.math.mod(u_ids, self._rank_size), tf.int32),
            self._rank_size,
            tf.constant([], tf.int64),
        )
        sc_all = hccl_ops.allgather(send_count, self._rank_size)
        local_ids = hccl_ops.all_to_all_v_c(send_data=u_ids, send_count_matrix=sc_all, rank=self._rank_id)
        local_cnts = hccl_ops.all_to_all_v_c(send_data=u_cnts, send_count_matrix=sc_all, rank=self._rank_id)

        local_ids = self._emb_table.count_filter.count_and_filter(local_ids, local_cnts)

        return local_ids, sc_all, u_idx

    def _embedding_all2all(
        self,
        emb: tf.Tensor,
        lookup_params: MPLookupParams,
        name: str,
        is_bp: bool = False,
    ) -> tf.Tensor:
        if self._rank_size > 1:
            emb_all2all_matrix = lookup_params.send_count_matrix * self._emb_table.dim
            if not is_bp:
                emb_all2all_matrix = tf.transpose(emb_all2all_matrix)
            emb = hccl_ops.all_to_all_v_c(send_data=emb, send_count_matrix=emb_all2all_matrix, rank=self._rank_id)

        emb = tf.reshape(emb, shape=(-1, self._emb_table.dim), name=name)
        return emb
