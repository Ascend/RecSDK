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

from mx_rec.util.communication.hccl_ops import get_rank_size
from mx_rec.util.ops import import_host_pipeline_ops
from mx_rec.core.asc.helper import get_asc_insert_func
from mx_rec.util.initialize import ConfigInitializer

from dataset import generate_dataset
from utils import FeatureSpecIns, create_feature_spec_list


def input_fn(params, create_fs_params, cfg, is_eval=False, use_one_shot=False):
    dataset = generate_dataset(cfg,
                               use_timestamp=params.use_timestamp,
                               batch_number=params.max_data_generate_steps * get_rank_size())

    if not params.modify_graph:
        feature_spec_list = create_feature_spec_list(create_fs_params.get("cfg"),
                                                     create_fs_params.get("use_timestamp"),
                                                     create_fs_params.get("use_multi_lookup"),
                                                     create_fs_params.get("multi_lookup_times"))
        if is_eval:
            FeatureSpecIns.get_instance().set_eval_feature_spec_list(feature_spec_list)
            dataset = dataset.map(get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=False))
            channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(
                False)
            import_host_pipeline_ops().clear_channel(channel_id)
        else:
            FeatureSpecIns.get_instance().set_train_feature_spec_list(feature_spec_list)
            dataset = dataset.map(get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=True))

    dataset = dataset.prefetch(100)

    if not use_one_shot:
        return dataset

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch
