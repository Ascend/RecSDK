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

"""
sparse ops
"""
from __future__ import absolute_import
import tensorflow as tf
from npu_bridge.hccl import hccl_ops
from sparse_ops import utils
from mpi4py import MPI

MPI.Init_thread(MPI.THREAD_MULTIPLE)  # must before emb_cache
utils.init = True


class SparseOps:
    """
    embedding相关的接口
    """

    def __init__(self, fallback=False):
        # context
        self.fallback = fallback
        self.all2all = hccl_ops.all_to_all_v

    def get_a2a_args(self, lookup_vec_size, mini_bs_w_field, rank_size, send_count, emb_vec_size):
        """
        获取a2a args信息
        """
        if self.fallback:
            send_count = tf.cond(lookup_vec_size > send_count * rank_size,
                                 lambda: mini_bs_w_field // rank_size,
                                 lambda: send_count)
        all2all_args = {
            "sc": tf.cast([send_count * emb_vec_size] * rank_size, tf.int64),
            "ss": tf.cast([send_count * emb_vec_size * i for i in range(rank_size)], tf.int64)}
        all2all_args['rc'] = all2all_args['sc']
        all2all_args['rs'] = all2all_args['ss']
        return all2all_args, send_count * rank_size

    def forward_alltoall(self, all2all_args, restore_vec, hot_pos, emb_vec, emb_vec_size):
        """
         emb的前向通信
         all2all_args：用all2all用到的参数
         restore_vec：恢复向量
         emb_vec：输入的emb
         """
        emb_vec = tf.reshape(emb_vec, [-1])

        result = self.all2all(send_data=emb_vec,
                              send_counts=all2all_args['sc'],
                              send_displacements=all2all_args['ss'],
                              recv_counts=all2all_args['rc'],
                              recv_displacements=all2all_args['rs']
                              )

        result = tf.reshape(result,
                            [-1, emb_vec_size],
                            name="after_all2all_reshape")
        if hot_pos is not None:
            result = tf.concat([tf.gather(result, hot_pos, name="hot_pos"), result], axis=0)

        output = tf.gather(result, restore_vec)
        return output

    def forward_alltoallc(self, all2all_args, restore_vec, emb_vec, emb_vec_size, rank):
        """
         emb的前向通信
         all2all_args：用all2all用到的参数
         restore_vec：恢复向量
         emb_vec：输入的emb
         """
        emb_vec = tf.reshape(emb_vec, [-1])

        result = hccl_ops.all_to_all_v_c(send_data=emb_vec,
                                         send_count_matrix=all2all_args,
                                         rank=rank
                                         )

        result = tf.reshape(result,
                            [-1, emb_vec_size],
                            name="after_all2all_reshape")
        output = tf.gather(result, restore_vec)
        return output

    def backward_alltoall(self, emb_grad, hot_pos, segment_ids, num_segments, all2all_args):
        """
         emb梯度的反向通信
         id_emb_grad：原始梯度
         segment_ids：恢复向量
         num_segments：压缩后的长度
         """
        # unique_local_grad 2node shape 37755 same with rc total and num_segment
        # unique_local_grad shape is [40052, 80]
        if hot_pos is not None:
            unique_local_grad = tf.math.unsorted_segment_sum(emb_grad,
                                                             segment_ids=segment_ids,
                                                             num_segments=num_segments + tf.shape(hot_pos)[0],
                                                             name="backward_combine")
            hot, cold = tf.split(unique_local_grad,
                                 [tf.shape(hot_pos)[0], tf.shape(unique_local_grad)[0] - tf.shape(hot_pos)[0]], axis=0)
            unique_local_grad = tf.tensor_scatter_nd_update(cold, tf.expand_dims(hot_pos, 1), hot)
        else:
            unique_local_grad = tf.math.unsorted_segment_sum(emb_grad,
                                                             segment_ids=segment_ids,
                                                             num_segments=num_segments, name="backward_combine")

        unique_grad = self.all2all(send_data=unique_local_grad,
                                   send_counts=all2all_args['rc'],
                                   send_displacements=all2all_args['rs'],
                                   recv_counts=all2all_args['sc'],
                                   recv_displacements=all2all_args['ss']
                                   )
        return unique_grad

    def backward_alltoallc(self, emb_grad, segment_ids, num_segments, all2all_args, rank):
        """
         emb梯度的反向通信
         id_emb_grad：原始梯度
         segment_ids：恢复向量
         num_segments：压缩后的长度
         """
        unique_local_grad = tf.math.unsorted_segment_sum(emb_grad,
                                                         segment_ids=segment_ids,
                                                         num_segments=num_segments, name="backward_combine")
        # unique_local_grad 2node shape 37755 same with rc total and num_segment
        # unique_local_grad shape is [40052, 80]
        unique_local_grad = tf.reshape(unique_local_grad, [-1])

        all2all_args = tf.transpose(all2all_args)
        unique_grad = hccl_ops.all_to_all_v_c(send_data=unique_local_grad,
                                              send_count_matrix=all2all_args,
                                              rank=rank
                                              )
        return unique_grad
