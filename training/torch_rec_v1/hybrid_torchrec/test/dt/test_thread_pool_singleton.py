#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
sys.modules['torch_npu'] = MagicMock

import pytest
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import (InputDistThreadPoolExecutorSingleton,
                                                                     DEFAULT_INPUT_DIST_THREADS,
                                                                     MAX_INPUT_DIST_THREADS)
from hybrid_torchrec.distributed.sharding.post_input_dist import (ThreadPoolExecutorSingleton,
                                                                  DEFAULT_POST_INPUT_THREADS,
                                                                  MAX_POST_INPUT_THREADS)


class TestThreadPoolExecutorSingleton:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """每次测试用例完成后重置对象"""
        yield
        InputDistThreadPoolExecutorSingleton._instance = None
        ThreadPoolExecutorSingleton._instance = None

    @pytest.mark.parametrize("singleton", [InputDistThreadPoolExecutorSingleton, ThreadPoolExecutorSingleton])
    def test_singleton_instance(self, singleton):
        instance1 = singleton()
        instance2 = singleton()
        assert instance1 is instance2
        assert isinstance(instance1.executor, ThreadPoolExecutor)

    @pytest.mark.parametrize("singleton", [InputDistThreadPoolExecutorSingleton, ThreadPoolExecutorSingleton])
    def test_with_valid_env_threads(self, monkeypatch, singleton):
        monkeypatch.setenv("INPUT_DIST_THREADS", str(MAX_INPUT_DIST_THREADS - 1))
        monkeypatch.setenv("POST_INPUT_THREADS", str(MAX_POST_INPUT_THREADS - 1))
        instance = singleton()
        assert instance.executor._max_workers in (MAX_INPUT_DIST_THREADS - 1, MAX_POST_INPUT_THREADS - 1)

    @pytest.mark.parametrize("singleton", [InputDistThreadPoolExecutorSingleton, ThreadPoolExecutorSingleton])
    def test_with_default_threads(self, singleton):
        instance = singleton()
        assert instance.executor._max_workers in (DEFAULT_INPUT_DIST_THREADS, DEFAULT_POST_INPUT_THREADS)

    @pytest.mark.parametrize("singleton", [InputDistThreadPoolExecutorSingleton, ThreadPoolExecutorSingleton])
    @pytest.mark.parametrize("invalid_count", [-1, 0, str(MAX_INPUT_DIST_THREADS + 1), "invalid"])
    def test_invalid_thread_counts(self, monkeypatch, singleton, invalid_count):
        monkeypatch.setenv("INPUT_DIST_THREADS", str(invalid_count))
        monkeypatch.setenv("POST_INPUT_THREADS", str(invalid_count))
        with pytest.raises((ValueError, Exception)):
            singleton()
