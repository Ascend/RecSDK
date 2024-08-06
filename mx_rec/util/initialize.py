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

import atexit
import dataclasses
import json

from mx_rec.constants.constants import MAX_INT32, GET_CONFIG_INSTANCE_ERR_MSG
from mx_rec.util.config_utils.embedding_utils import SparseEmbedConfig
from mx_rec.util.config_utils.feature_spec_utils import FeatureSpecConfig
from mx_rec.util.config_utils.hybrid_mgmt_utils import HybridManagerConfig
from mx_rec.util.config_utils.optimizer_utils import OptimizerConfig
from mx_rec.util.config_utils.train_param import TrainParamsConfig
from mx_rec.util.framework_npu_env.tfa_env import set_ascend_env
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.log import logger
from mx_rec.util.perf_factory.bind_cpu import bind_cpu
from mx_rec.validator.validator import para_checker_decorator, ClassValidator, \
    IntValidator, ValueCompareValidator


class ConfigInitializer:
    _single_instance = None

    @para_checker_decorator(check_option_list=[
        ("max_steps", IntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("train_steps", IntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("eval_steps", IntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("save_steps", IntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        (["max_steps", "train_steps", "eval_steps"], ValueCompareValidator, {"target": 0},
         ["check_at_least_one_not_equal_to_target"]),
        ("if_load", ClassValidator, {"classes": (bool,)}),
        ("use_dynamic", ClassValidator, {"classes": (bool,)}),
        ("use_dynamic_expansion", ClassValidator, {"classes": (bool,)}),
        ("bind_cpu", ClassValidator, {"classes": (bool,)}),
        ("save_checkpoint_due_time", IntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("save_delta_checkpoints_secs", IntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("is_incremental_checkpoint", ClassValidator, {"classes": (bool,)}),
        ("restore_model_version", ClassValidator, {"classes": (str,)}),
    ])
    @bind_cpu
    def __init__(self, **kwargs):
        self._modify_graph = False

        self._max_steps = kwargs.get("max_steps", -1)
        self._train_steps = kwargs.get("train_steps", -1)
        self._eval_steps = kwargs.get("eval_steps", -1)
        self._save_steps = kwargs.get("save_steps", -1)

        self._if_load = kwargs.get("if_load", False)

        self._use_static = not kwargs.get("use_dynamic", True)
        self._use_dynamic_expansion = kwargs.get("use_dynamic_expansion", False)

        self._is_terminated = False

        self._sparse_embed_config = SparseEmbedConfig()
        self._feature_spec_config = FeatureSpecConfig()
        self._hybrid_manager_config = HybridManagerConfig()
        self._optimizer_config = OptimizerConfig()
        self._train_params_config = TrainParamsConfig()

        # incremental checkpoint settings
        self._save_checkpoint_due_time = kwargs.get("save_checkpoint_due_time", None)
        self._save_delta_checkpoints_secs = kwargs.get("save_delta_checkpoints_secs", None)
        self._is_incremental_checkpoint = kwargs.get("is_incremental_checkpoint", False)
        self._restore_model_version = kwargs.get("restore_model_version", None)
        self._recent_key_count_threshold = kwargs.get("recent_key_count_threshold", 0)
        self._is_first_restore = True

    @property
    def save_checkpoint_due_time(self):
        return self._save_checkpoint_due_time

    @property
    def save_delta_checkpoints_secs(self):
        return self._save_delta_checkpoints_secs

    @property
    def is_incremental_checkpoint(self):
        return self._is_incremental_checkpoint

    @property
    def restore_model_version(self):
        return self._restore_model_version

    @property
    def is_first_restore(self):
        return self._is_first_restore

    @is_first_restore.setter
    def is_first_restore(self, is_first_restore: bool):
        self._is_first_restore = is_first_restore

    @property
    def modify_graph(self):
        return self._modify_graph

    @modify_graph.setter
    def modify_graph(self, modify_graph):
        self._modify_graph = modify_graph

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, step: int):
        self._max_steps = step

    @property
    def train_steps(self):
        return self._train_steps

    @train_steps.setter
    def train_steps(self, step: int):
        self._train_steps = step

    @property
    def eval_steps(self):
        return self._eval_steps

    @property
    def save_steps(self):
        return self._save_steps

    @property
    def if_load(self):
        return self._if_load

    @property
    def use_static(self):
        return self._use_static

    @property
    def use_dynamic_expansion(self):
        return self._use_dynamic_expansion

    @property
    def sparse_embed_config(self):
        return self._sparse_embed_config

    @sparse_embed_config.setter
    def sparse_embed_config(self, sparse_emb_config_instance):
        self._sparse_embed_config = sparse_emb_config_instance

    @property
    def feature_spec_config(self):
        return self._feature_spec_config

    @feature_spec_config.setter
    def feature_spec_config(self, feature_spec_config_instance):
        self._feature_spec_config = feature_spec_config_instance

    @property
    def hybrid_manager_config(self):
        return self._hybrid_manager_config

    @hybrid_manager_config.setter
    def hybrid_manager_config(self, hybrid_manager_config_instance):
        self._hybrid_manager_config = hybrid_manager_config_instance

    @property
    def optimizer_config(self):
        return self._optimizer_config

    @optimizer_config.setter
    def optimizer_config(self, optimizer_config_instance):
        self._optimizer_config = optimizer_config_instance

    @property
    def train_params_config(self):
        return self._train_params_config

    @train_params_config.setter
    def train_params_config(self, train_params_config_instance):
        self._train_params_config = train_params_config_instance

    @eval_steps.setter
    def eval_steps(self, steps):
        self._eval_steps = steps

    @save_steps.setter
    def save_steps(self, steps):
        self._save_steps = steps

    @if_load.setter
    def if_load(self, flag):
        self._if_load = flag

    @staticmethod
    def get_instance():
        if ConfigInitializer._single_instance is None:
            raise EnvironmentError(GET_CONFIG_INSTANCE_ERR_MSG)

        return ConfigInitializer._single_instance

    @staticmethod
    def set_instance(**kwargs):
        if ConfigInitializer._single_instance is not None:
            raise EnvironmentError("ConfigInitializer has been initialized once, twice initialization was forbidden.")

        ConfigInitializer._single_instance = ConfigInitializer(**kwargs)

    def terminate(self):
        logger.info("python process run into terminate")
        if self._is_terminated:
            logger.warning("The initializer has already been released once, please do not release it again.")
            return

        if self._hybrid_manager_config.asc_manager is not None:
            self._hybrid_manager_config.del_asc_manager()
        logger.info("python process run terminate success")

        self._is_terminated = True


def init(**kwargs):
    logger.info("The environment variables set for mxRec is: %s",
                json.dumps(dataclasses.asdict(global_env), ensure_ascii=False))
    from mpi4py import MPI
    set_ascend_env()
    ConfigInitializer.set_instance(**kwargs)
    atexit.register(terminate_config_initializer)


def terminate_config_initializer():
    try:
        ConfigInitializer.get_instance().terminate()
    except EnvironmentError as err:
        if GET_CONFIG_INSTANCE_ERR_MSG not in str(err):
            raise err
        logger.warning(GET_CONFIG_INSTANCE_ERR_MSG)
