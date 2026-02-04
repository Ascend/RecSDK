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
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode
from torchrec import DataType
from torchrec.distributed.types import CacheParams, KeyValueParams
from torchrec.modules.embedding_configs import ShardingType

from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints


class TestDynamicEmbParameterConstraints:
    @staticmethod
    def test_ok():
        try:
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_use_dynamicemb_only_support_true():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                use_dynamicemb=False,
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
            )

    @staticmethod
    def test_dynamicemb_options_type_err():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                dynamicemb_options="xxx",
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
            )

    @staticmethod
    @pytest.mark.parametrize(
        "invalid_value",
        [
            None,
            "xxx",
            [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value],
            [ShardingType.ROW_WISE.value] * 10001,
        ],
    )
    def test_sharding_types_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=invalid_value,
                compute_kernels=["fused"],
            )

    @staticmethod
    @pytest.mark.parametrize("invalid_value", [None, "xxx", ["fused", "dense"], ["fused"] * 10001])
    def test_compute_kernels_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=invalid_value,
            )

    @staticmethod
    def test_only_support_row_wise_fused():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["dense"],
            )

    @staticmethod
    def test_min_partition_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                min_partition=1,
            )

    @staticmethod
    def test_pooling_factors_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                pooling_factors=[2.0],
            )

    @staticmethod
    def test_num_poolings_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                num_poolings=[1.0],
            )

    @staticmethod
    def test_batch_sizes_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                batch_sizes=[32],
            )

    @staticmethod
    def test_is_weighted_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                is_weighted=True,
            )

    @staticmethod
    def test_cache_params_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                cache_params=CacheParams(),
            )

    @staticmethod
    def test_enforce_hbm_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                enforce_hbm=True,
            )

    @staticmethod
    def test_stochastic_rounding_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                stochastic_rounding=True,
            )

    @staticmethod
    def test_bounds_check_mode_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                bounds_check_mode=BoundsCheckMode.IGNORE,
            )

    @staticmethod
    def test_feature_names_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                feature_names=["xxx"],
            )

    @staticmethod
    def test_output_dtype_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                output_dtype=DataType.FP32,
            )

    @staticmethod
    def test_device_group_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                device_group="xxx",
            )

    @staticmethod
    def test_key_value_params_only_support_default_value():
        with pytest.raises(ValueError):
            DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                key_value_params=KeyValueParams(),
            )
