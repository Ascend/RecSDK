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

import pytest
import torch

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbPoolingMode,
    DynamicEmbInitializerMode,
    DynamicEmbInitializerArgs,
    DynamicEmbScoreStrategy,
    DynamicEmbCheckMode,
    DynamicEmbTableOptions,
    DynamicEmbEvictStrategy,
    DistType,
    next_power_of_2,
    dyn_emb_to_torch,
    torch_to_dyn_emb,
    create_dynamicemb_table,
    get_optimizer_state_dim,
)
from dynamic_emb_extensions import InitializerArgs, DynamicEmbDataType, DynamicEmbTable, OptimizerType, SafeCheckMode


class TestDynamicEmbPoolingMode:
    @staticmethod
    def test_ok():
        assert DynamicEmbPoolingMode.SUM.value == 0
        assert DynamicEmbPoolingMode.MEAN.value == 1
        assert DynamicEmbPoolingMode.NONE.value == 2


class TestDynamicEmbInitializerMode:
    @staticmethod
    def test_ok():
        assert DynamicEmbInitializerMode.NORMAL.value == "normal"
        assert DynamicEmbInitializerMode.TRUNCATED_NORMAL.value == "truncated_normal"
        assert DynamicEmbInitializerMode.UNIFORM.value == "uniform"
        assert DynamicEmbInitializerMode.CONSTANT.value == "constant"
        assert DynamicEmbInitializerMode.DEBUG.value == "debug"


class TestDynamicEmbEvictStrategy:
    @staticmethod
    def test_ok():
        assert DynamicEmbEvictStrategy.LRU.value.value == 0
        assert DynamicEmbEvictStrategy.LFU.value.value == 1
        assert DynamicEmbEvictStrategy.EPOCH_LRU.value.value == 2
        assert DynamicEmbEvictStrategy.EPOCH_LFU.value.value == 3
        assert DynamicEmbEvictStrategy.CUSTOMIZED.value.value == 4


class TestDynamicEmbInitializerArgs:
    @staticmethod
    def test_init_ok():
        init_args = DynamicEmbInitializerArgs()
        assert isinstance(init_args.as_ctype(), InitializerArgs)

    @staticmethod
    def test_mode_type_err():
        with pytest.raises(ValueError):
            DynamicEmbInitializerArgs(mode="xxx")

    @staticmethod
    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
    def test_mean_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbInitializerArgs(mean=invalid_value)

    @staticmethod
    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
    def test_std_dev_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbInitializerArgs(std_dev=invalid_value)

    @staticmethod
    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
    def test_lower_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbInitializerArgs(lower=invalid_value)

    @staticmethod
    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
    def test_upper_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbInitializerArgs(upper=invalid_value)

    @staticmethod
    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
    def test_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbInitializerArgs(value=invalid_value)

    @staticmethod
    def test_eq_isinstance_err():
        init_args = DynamicEmbInitializerArgs()
        other_args = "xxxx"
        assert init_args.__eq__(other_args) is NotImplementedError

    @staticmethod
    def test_eq_normal_ok():
        init_args = DynamicEmbInitializerArgs()
        other_args = DynamicEmbInitializerArgs()
        assert init_args.__eq__(other_args)

    @staticmethod
    def test_eq_not_normal_ok():
        init_args = DynamicEmbInitializerArgs()
        other_args = DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.CONSTANT)
        assert init_args.__eq__(other_args)

    @staticmethod
    def test_ne_isinstance_err():
        init_args = DynamicEmbInitializerArgs()
        other_args = "xxx"
        assert init_args.__ne__(other_args) is NotImplementedError

    @staticmethod
    def test_ne_ok():
        init_args = DynamicEmbInitializerArgs()
        other_args = DynamicEmbInitializerArgs(mean=0.9)
        assert init_args.__ne__(other_args)


class TestDynamicEmbScoreStrategy:
    @staticmethod
    def test_ok():
        assert DynamicEmbScoreStrategy.TIMESTAMP == 0
        assert DynamicEmbScoreStrategy.STEP == 1
        assert DynamicEmbScoreStrategy.CUSTOMIZED == 2
        assert DynamicEmbScoreStrategy.LFU == 3


class TestDynamicEmbCheckMode:
    @staticmethod
    def test_ok():
        assert DynamicEmbCheckMode.ERROR == 0
        assert DynamicEmbCheckMode.WARNING == 1
        assert DynamicEmbCheckMode.IGNORE == 2


class TestDynamicEmbTableOptions:
    @staticmethod
    def test_init_ok():
        try:
            DynamicEmbTableOptions()
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_training_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(training=-1)

    @staticmethod
    def test_initializer_args_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(initializer_args=-1)

    @staticmethod
    def test_eval_initializer_args_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(eval_initializer_args=-1)

    @staticmethod
    def test_eval_initializer_args_not_normal_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(
                eval_initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.CONSTANT)
            )

    @staticmethod
    def test_init_capacity_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(init_capacity="xxx")

    @staticmethod
    def test_init_capacity_default_value_ok():
        try:
            DynamicEmbTableOptions(init_capacity=0)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_init_capacity_power_2_ok():
        assert DynamicEmbTableOptions(init_capacity=3).init_capacity == 4

    @staticmethod
    def test_init_capacity_value_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(init_capacity=-1)

    @staticmethod
    def test_max_load_factor_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(max_load_factor="xxx")

    @staticmethod
    def test_max_load_factor_value_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(max_load_factor=0.0)

    @staticmethod
    def test_score_strategy_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(score_strategy=-1)

    @staticmethod
    def test_bucket_capacity_ok():
        assert DynamicEmbTableOptions(bucket_capacity=31).bucket_capacity == 32

    @staticmethod
    def test_safe_check_mode_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(safe_check_mode=-1)

    @staticmethod
    def test_global_hbm_for_values_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(global_hbm_for_values=-1)

    @staticmethod
    def test_caching_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(caching=-1)

    @staticmethod
    def test_caching_should_be_false_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(caching=True)

    @staticmethod
    def test_external_storage_value_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(external_storage=-1)

    @staticmethod
    def test_index_type_type_err():
        with pytest.raises(ValueError):
            DynamicEmbTableOptions(index_type=torch.float32)

    @staticmethod
    def test_eq_isinstance_err():
        de_table_opt = DynamicEmbTableOptions()
        other_table_opt = "xxx"
        assert de_table_opt.__eq__(other_table_opt) is NotImplementedError

    @staticmethod
    def test_eq_ok():
        de_table_opt = DynamicEmbTableOptions()
        other_table_opt = DynamicEmbTableOptions()
        assert de_table_opt.__eq__(other_table_opt)

    @staticmethod
    def test_ne_isinstance_err():
        de_table_opt = DynamicEmbTableOptions()
        other_table_opt = "xxx"
        assert de_table_opt.__ne__(other_table_opt) is NotImplementedError

    @staticmethod
    def test_ne_ok():
        de_table_opt = DynamicEmbTableOptions()
        other_table_opt = DynamicEmbTableOptions()
        assert not de_table_opt.__ne__(other_table_opt)


class TestDistType:
    @staticmethod
    def test_ok():
        assert DistType.CONTINUOUS.value == "continuous"
        assert DistType.ROUNDROBIN.value == "roundrobin"


class TestNextPowerOf2:
    @staticmethod
    def test_n_is_0():
        assert next_power_of_2(0) == 1

    @staticmethod
    def test_n_is_already_a_power_of_2():
        assert next_power_of_2(2) == 2

    @staticmethod
    def test_n_is_3():
        assert next_power_of_2(3) == 4


class TestDynEmbToTorch:
    @staticmethod
    def test_ok():
        assert dyn_emb_to_torch(DynamicEmbDataType.Float32) == torch.float32
        assert dyn_emb_to_torch(DynamicEmbDataType.BFloat16) == torch.bfloat16
        assert dyn_emb_to_torch(DynamicEmbDataType.Float16) == torch.float16
        assert dyn_emb_to_torch(DynamicEmbDataType.Int64) == torch.int64
        assert dyn_emb_to_torch(DynamicEmbDataType.Int32) == torch.int32
        assert dyn_emb_to_torch(DynamicEmbDataType.Size_t) == torch.int64
        with pytest.raises(ValueError):
            dyn_emb_to_torch("xxx")


class TestTorchToDynEmb:
    @staticmethod
    def test_ok():
        assert torch_to_dyn_emb(torch.float32) == DynamicEmbDataType.Float32
        assert torch_to_dyn_emb(torch.bfloat16) == DynamicEmbDataType.BFloat16
        assert torch_to_dyn_emb(torch.float16) == DynamicEmbDataType.Float16
        assert torch_to_dyn_emb(torch.int64) == DynamicEmbDataType.Int64
        assert torch_to_dyn_emb(torch.int32) == DynamicEmbDataType.Int32
        with pytest.raises(ValueError):
            torch_to_dyn_emb("xxx")


class TestCreateDynamicEmbTable:
    @staticmethod
    def test_ok():
        table_options = DynamicEmbTableOptions(training=False)
        # mock inferred from the context
        table_options.index_type = torch.int64
        table_options.embedding_dtype = torch.float32
        table_options.evict_strategy = DynamicEmbEvictStrategy.LRU
        table_options.dim = 8
        table_options.init_capacity = 128
        table_options.max_capacity = 256
        table_options.local_hbm_for_values = 1024
        table_options.bucket_capacity = 128
        table_options.max_load_factor = 0.8
        table_options.block_size = 128
        table_options.io_block_size = 1024
        table_options.device_id = 0
        table_options.io_by_cpu = False
        table_options.use_constant_memory = False
        table_options.reserved_key_start_bit = 0
        table_options.num_of_buckets_per_alloc = 1
        table_options.initializer_args = DynamicEmbInitializerArgs()
        table_options.safe_check_mode = SafeCheckMode.IGNORE
        table_options.optimizer_type = OptimizerType.Null
        assert isinstance(create_dynamicemb_table(table_options), DynamicEmbTable)


class TestGetOptimizerStateDim:
    @staticmethod
    def test_ok():
        assert get_optimizer_state_dim(OptimizerType.RowWiseAdaGrad, 8, torch.float32) == (16 // 4)
        assert get_optimizer_state_dim(OptimizerType.Adam, 8, torch.float32) == (8 * 2)
        assert get_optimizer_state_dim(OptimizerType.AdaGrad, 8, torch.float32) == 8
