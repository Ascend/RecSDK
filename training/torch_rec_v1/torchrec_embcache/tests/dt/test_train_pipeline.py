#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import patch, MagicMock
import pytest
import torch

from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.optim.keyed import CombinedOptimizer

from hybrid_torchrec.constants import MAX_LOCAL_UNIQUE_PARALLEL_BATCH_NUM
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist


class TestEmbCacheTrainPipeline:
    @pytest.mark.parametrize(
        "env_value, should_raise, error_pattern",
        [
            # 有效值
            ("1", False, None),
            ("2", False, None),
            (str(MAX_LOCAL_UNIQUE_PARALLEL_BATCH_NUM), False, None),
            # 无效值 - 小于最小值
            ("0", True, r"Param error: LOCAL_UNIQUE_PARALLEL_BATCH_NUM must be in \[1, \d+\], but got 0\."),
            ("-1", True, r"Param error, LOCAL_UNIQUE_PARALLEL_BATCH_NUM must be a numberbut got -1\."),
            # 无效值 - 大于最大值
            (
                str(MAX_LOCAL_UNIQUE_PARALLEL_BATCH_NUM + 1), 
                True, 
                r"Param error: LOCAL_UNIQUE_PARALLEL_BATCH_NUM must be in \[1, \d+\], but got \d+\."
            ),
            # 无效值 - 非数字
            ("abc", True, r"Param error, LOCAL_UNIQUE_PARALLEL_BATCH_NUM must be a numberbut got abc\."),
            ("1.5", True, r"Param error, LOCAL_UNIQUE_PARALLEL_BATCH_NUM must be a numberbut got 1.5\."),
            ("", True, r"Param error, LOCAL_UNIQUE_PARALLEL_BATCH_NUM must be a numberbut got \."),
        ],
    )
    def test_local_unique_parallel_batch_num_validation(self, env_value, should_raise, error_pattern):
        """测试LOCAL_UNIQUE_PARALLEL_BATCH_NUM环境变量的校验逻辑"""
        with patch.dict(os.environ, {"LOCAL_UNIQUE_PARALLEL_BATCH_NUM": env_value}):
            # Mock必要的依赖项
            mock_model = MagicMock(spec=DistributedModelParallel)
            mock_optimizer = MagicMock(spec=CombinedOptimizer)
            mock_cpu_device = torch.device("cpu")
            mock_npu_device = torch.device("cpu")  # 在测试环境中使用CPU模拟NPU
            
            if should_raise:
                with pytest.raises(ValueError, match=error_pattern):
                    EmbCacheTrainPipelineSparseDist(
                        model=mock_model,
                        optimizer=mock_optimizer,
                        cpu_device=mock_cpu_device,
                        npu_device=mock_npu_device,
                    )
            else:
                # 应该正常创建，不抛出异常
                pipeline = EmbCacheTrainPipelineSparseDist(
                    model=mock_model,
                    optimizer=mock_optimizer,
                    cpu_device=mock_cpu_device,
                    npu_device=mock_npu_device,
                )
                assert pipeline.local_unique_parallel_batch_num == int(env_value)