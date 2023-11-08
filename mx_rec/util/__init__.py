#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

__all__ = [
    "init", "get_rank_id", "get_initializer", "terminate_config_initializer", "clear_channel",
    "get_dense_and_sparse_variable", "set_if_load", "set_ascend_global_hashtable_collection",
    "get_ascend_global_hashtable_collection", "get_rank_size", "get_host_pipeline_ops",
    "get_use_dynamic_expansion", "set_ascend_table_name_must_contain", "get_target_batch"
]

from mx_rec.util.tf_version_adapter import npu_ops, hccl_ops, NPUCheckpointSaverHook
from mx_rec.util.initialize import init, get_rank_id, get_initializer, terminate_config_initializer, clear_channel, \
    set_if_load, set_ascend_global_hashtable_collection, get_ascend_global_hashtable_collection, get_rank_size, \
    get_host_pipeline_ops, get_use_dynamic_expansion, set_ascend_table_name_must_contain, get_target_batch
from mx_rec.util.variable import get_dense_and_sparse_variable
