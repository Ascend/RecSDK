#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import shutil
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import torch.nn
from torchrec_embcache.saver import Saver, SAVE_PATH_MAX_LEN, TIMESTAMP_FORMAT
from torchrec_embcache.utils import safe_makedirs


class TestSaver:
    @staticmethod
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_init_with_no_rank_should_ok(*mock):
        saver = Saver()
        assert saver.rank == 0

    @staticmethod
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=10)
    @pytest.mark.parametrize("rank", [0, 2, 9])
    def test_init_with_rank_should_ok(mock0, mock1, rank):
        _ = Saver(rank)

    @staticmethod
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=10)
    @pytest.mark.parametrize("rank", [-1, 10, 15])
    def test_init_with_exceed_rank_should_failed(mock0, mock1, rank):
        with pytest.raises(ValueError):
            _ = Saver(rank)

    @staticmethod
    @patch("torch.distributed.is_initialized", return_value=False)
    def test_init_with_no_rank_should_failed(*mock):
        with pytest.raises(ValueError):
            _ = Saver()

    @staticmethod
    @pytest.mark.parametrize("rank", ["rank_str", False, -1])
    def test_init_with_invalid_rank_should_failed(rank):
        with pytest.raises(ValueError):
            _ = Saver(rank)

    @staticmethod
    def test_save_with_invalid_path_or_module_should_failed():
        saver = Saver(0)
        with pytest.raises(TypeError):
            saver.save(None, 1)
        with pytest.raises(ValueError):
            path = "a" * (SAVE_PATH_MAX_LEN + 1)
            saver.save(None, path)
        with pytest.raises(ValueError):
            saver.save(None, "../../bin")
        with pytest.raises(ValueError):
            saver.save(None, "/usr/bin")
        with pytest.raises(ValueError):
            saver.save(None, "password")
        with pytest.raises(ValueError):
            module = torch.nn.Module()
            saver.save(module, "save_dir")

    @staticmethod
    def test_load_with_invalid_path_should_failed():
        saver = Saver(0)
        with pytest.raises(ValueError):
            # not exist directory
            saver.load(None, "xxx")

    @staticmethod
    def test_load_with_invalid_module_should_failed():
        saver = Saver(0)
        with pytest.raises(ValueError):
            # error module
            module = torch.nn.Module()
            saver.load(module, "save_dir")

    @staticmethod
    def test_load_with_ts_dir_not_exist_should_failed():
        saver = Saver(0)
        # don't have timestamp directory in path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        temp_dir = datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT) + str(random.randint(0, 100000))
        temp_dir = os.path.join(dir_path, temp_dir)
        safe_makedirs(temp_dir)
        with pytest.raises(ValueError):
            module = torch.nn.Module()
            saver.load(module, temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
