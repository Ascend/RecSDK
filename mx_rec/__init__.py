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

__version__ = "6.0.RC3"
__all__ = ["version", "__version__"]

from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from mx_rec.util.tf_version_adapter import npu_ops, hccl_ops, NPUCheckpointSaverHook
from mx_rec.saver.patch import patch_for_saver, patch_for_summary_writer
from mx_rec.graph.patch import patch_for_dataset, patch_for_chief_session_creator, patch_for_bool_gauge, \
    patch_for_assert_eval_spec, patch_for_scale_loss, patch_for_session
from mx_rec.data.patch import patch_for_dataset_eos_map
from mx_rec.optimizers.base import patch_for_optimizer
from mx_rec.saver.warm_start import patch_for_warm_start

patch_for_saver()
patch_for_summary_writer()
patch_for_dataset()
patch_for_dataset_eos_map()
patch_for_scale_loss()
patch_for_chief_session_creator()
patch_for_assert_eval_spec()
patch_for_bool_gauge()
patch_for_optimizer()
patch_for_session()
patch_for_warm_start()


def version():
    return __version__
