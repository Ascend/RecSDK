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

import tensorflow as tf

from mx_rec import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from mx_rec.util.config_utils.feature_spec_utils import FeatureSpecConfig
from mx_rec.util.config_utils.optimizer_utils import OptimizerConfig


class MockHybridManagerConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.freeze = kwargs.get("freeze", False)
        self.asc_manager = kwargs.get("asc_manager", None)

    def trigger_evict(self):
        return self.kwargs.get("trigger_evict", True)

    def set_asc_manager(self, cache):
        pass

    def save_host_data(self, root_dir, save_delta):
        pass

    def get_host_data(self, table_name):
        return self.kwargs.get("host_data", [])

    def restore_host_data(self, path):
        pass

    def get_load_offset(self, table_name):
        return self.kwargs.get("load_offset")


class MockSparseEmbedConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.table_instance_dict = kwargs.get("table_instance_dict", {})
        self.dangling_table = kwargs.get("dangling_table", [])
        self.table_name_set = kwargs.get("table_name_set", set())
        self.removing_var_list = kwargs.get("removing_var_list", [])

    @staticmethod
    def insert_dangling_table(table_name):
        pass

    @staticmethod
    def insert_removing_var_list(var_name):
        pass

    @staticmethod
    def insert_table_instance(name, key, instance, eval_flag):
        pass

    def get_table_instance(self, var):
        return self.kwargs.get("var", None)

    def get_table_instance_by_name(self, name):
        return self.kwargs.get("var", None)


class MockTrainParamsConfig:
    def __init__(self, **kwargs):
        def _get_training_mode_channel_id(is_training):
            _dict = {True: 0, False: 1}
            return _dict.get(is_training)

        def _insert_training_mode_channel_id(is_training):
            pass

        def _get_merged_multi_lookup(is_training):
            return kwargs.get("merged_multi_lookup", False)

        def _insert_merged_multi_lookup(is_training, flag):
            pass

        def _set_initializer(is_training, initializer):
            pass

        def _set_target_batch(is_training, batch):
            pass

        self.ascend_global_hashtable_collection = kwargs.get(
            "ascend_global_hashtable_collection", ASCEND_GLOBAL_HASHTABLE_COLLECTION
        )
        self.is_graph_modify_hook_running = kwargs.get("is_graph_modify_hook_running", True)
        self.experimental_mode = kwargs.get("experimental_mode")
        self.bool_gauge_set = kwargs.get("bool_gauge_set", [])
        self.iterator_type = kwargs.get("iterator_type", "")
        self.sparse_dir = kwargs.get("sparse_dir", "")
        self.dataset_element_spec = kwargs.get("dataset_element_spec", {})

        self.get_training_mode_channel_id = _get_training_mode_channel_id
        self.insert_training_mode_channel_id = _insert_training_mode_channel_id
        self.get_merged_multi_lookup = _get_merged_multi_lookup
        self.insert_merged_multi_lookup = _insert_merged_multi_lookup
        self.set_initializer = _set_initializer
        self.set_target_batch = _set_target_batch


class MockConfigInitializer:
    """
    原始ConfigInitializer的mock
    """

    def __init__(self, **kwargs):
        self.use_dynamic_expansion = kwargs.get("use_dynamic_expansion", False)
        self.use_static = kwargs.get("use_static", False)
        self.modify_graph = kwargs.get("modify_graph", True)
        self.max_steps = kwargs.get("max_steps", -1)
        self.train_steps = kwargs.get("get_train_steps", -1)
        self.eval_steps = kwargs.get("eval_steps", -1)
        self.save_steps = kwargs.get("save_steps", -1)
        self.if_load = kwargs.get("if_load", False)
        self.iterator_type = kwargs.get("iterator_type", "MakeIterator")
        self.sparse_dir = kwargs.get("sparse_dir", "")
        self.is_incremental_checkpoint = kwargs.get("is_incremental_checkpoint", False)
        self.restore_model_version = kwargs.get("restore_model_version")

        self.hybrid_manager_config = MockHybridManagerConfig(**kwargs)
        self.sparse_embed_config = MockSparseEmbedConfig(**kwargs)
        self.train_params_config = MockTrainParamsConfig(**kwargs)
        self.optimizer_config = OptimizerConfig()
        self.feature_spec_config = FeatureSpecConfig()

        self.use_lccl = False

    def get_instance(self):
        return self


class MockGlobalEnv:
    def __init__(self, **kwargs):
        self.tf_device = kwargs.get("tf_device", "NPU")
        self.rank_table_file = kwargs.get("rank_table_file", "")


class MockSparseEmbedding:
    """
    原始SparseEmbedding会调用很多接口，用MockSparseEmbedding防止mock过多接口
    """

    def __init__(
        self,
        table_name="test_table",
        slice_device_vocabulary_size=10,
        embedding_size=5,
        init_param=1.0,
        emb_initializer=tf.zeros_initializer(),
    ):
        self.is_hbm = True
        self.is_dp = False
        self.is_grad = True
        self.table_name = table_name
        self.slice_device_vocabulary_size = slice_device_vocabulary_size
        self.embedding_size = tf.TensorShape([embedding_size])
        self.init_param = init_param
        self.emb_initializer = emb_initializer
        self.padding_keys = [666]
        self.padding_keys_len = 4096
        self.padding_keys_mask = True
        self.send_count = 4096
        self.variable = tf.compat.v1.get_variable(
            table_name,
            shape=[slice_device_vocabulary_size, embedding_size],
            trainable=False,
            initializer=tf.ones_initializer(),
        )


class MockHostPipeLineOps:
    """
    用于mock host_pipeline_ops，返回静态/动态 readEmbKey
    """

    def __init__(self):
        def _mock_read_emb_key_v2_fn(concat_tensor, **kwargs):
            return 0

        def _mock_read_emb_key_v2_dynamic_fn(concat_tensor, tensorshape_split_list, **kwargs):
            return 1

        self.read_emb_key_v2 = _mock_read_emb_key_v2_fn
        self.read_emb_key_v2_dynamic = _mock_read_emb_key_v2_dynamic_fn


class MockHcclOps:
    """
    用于mock hccl_ops
    """

    def __init__(self, shape=None):
        def _mock_all_to_all_v(send_data, send_counts, send_displacements, recv_counts, recv_displacements):
            if shape is None:
                return tf.constant(1, dtype=tf.int64, name="all_to_all_v")
            return tf.ones(shape, dtype=tf.int64, name="all_to_all_v")

        def _mock_all_to_all_v_c(send_data, send_count_matrix, rank):
            if shape is None:
                return tf.constant(1, dtype=tf.int64, name="all_to_all_v_c")
            return tf.ones(shape, dtype=tf.int64, name="all_to_all_v_c")

        self.all_to_all_v = _mock_all_to_all_v
        self.all_to_all_v_c = _mock_all_to_all_v_c


class MockOptimizer:
    """
    用于mock optimizer
    """

    def __init__(self):
        self.slot_num = 2
        self.derivative = 2

    def get_slot_init_values(self):
        initial_momentum_value = 0.0
        initial_velocity_value = 0.0
        return [initial_momentum_value, initial_velocity_value]

    def update_op(self, optimizer, g):
        pass

    def _apply_spare_duplicate_indices(self, grad, var):
        return self._apply_sparse(grad, var)

    def _apply_sparse(self, grad, var):
        pass

    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    def _apply_dense(self, grad, var):
        pass

    def _apply_sparse_duplicate_indices(self, grad, var):
        return self._apply_sparse(grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
        return self._resource_apply_sparse(grad, handle, indices)


class MockAscManager:
    """
    用于mock get_asc_manager()
    """

    def __init__(self):
        def _mock_get_table_size(self):
            return 0

        def _mock_get_table_capacity(self):
            return 1

        self.get_table_size = _mock_get_table_size
        self.get_table_capacity = _mock_get_table_capacity


class MockHybridMgmt:
    """
    用于mock HybridMgmt()
    """

    def __init__(self, is_initialized=True):
        def _mock_initialize(
            rank_info=0, emb_info=1, if_load=False, threshold_values=3, is_incremental_checkpoint=False, use_lccl=False
        ):
            return is_initialized

        self.initialize = _mock_initialize


class MockFeatureSpec:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.table_name = kwargs.get("table_name")
