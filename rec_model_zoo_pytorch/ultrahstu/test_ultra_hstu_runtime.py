#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import sys
import unittest
from pathlib import Path

import torch


os.environ["ULTRA_HSTU_DISABLE_AUTO_INSTALL"] = "1"
patches_path = str(Path(__file__).resolve().parents[1] / "patched")
if patches_path not in sys.path:
    sys.path.append(patches_path)

from ultra_hstu_runtime import build_ultra_attention_mask  # noqa: E402


class UltraHSTURuntimeTest(unittest.TestCase):
    def test_full_window_matches_causal_valid_mask(self) -> None:
        lengths = torch.tensor([5, 3], dtype=torch.long)
        mask = build_ultra_attention_mask(
            lengths=lengths,
            max_seq_len=5,
            local_window=5,
            global_window=0,
            truncation_length=0,
            layer_idx=0,
            num_full_layers=0,
        )
        positions = torch.arange(5)
        causal = positions.view(5, 1) >= positions.view(1, 5)
        self.assertTrue(torch.equal(mask[0], causal))
        self.assertTrue(torch.equal(mask[1, :3, :3], causal[:3, :3]))
        self.assertFalse(mask[1, 3:].any())
        self.assertFalse(mask[1, :, 3:].any())

    def test_semi_local_attention_unions_local_and_global_keys(self) -> None:
        lengths = torch.tensor([5], dtype=torch.long)
        mask = build_ultra_attention_mask(
            lengths=lengths,
            max_seq_len=5,
            local_window=2,
            global_window=1,
            truncation_length=0,
            layer_idx=0,
            num_full_layers=0,
        )
        self.assertEqual(torch.nonzero(mask[0, 4], as_tuple=False).flatten().tolist(), [0, 3, 4])
        self.assertEqual(torch.nonzero(mask[0, 2], as_tuple=False).flatten().tolist(), [0, 1, 2])

    def test_attention_truncation_keeps_latest_suffix(self) -> None:
        lengths = torch.tensor([5], dtype=torch.long)
        mask = build_ultra_attention_mask(
            lengths=lengths,
            max_seq_len=5,
            local_window=5,
            global_window=1,
            truncation_length=2,
            layer_idx=1,
            num_full_layers=1,
        )
        self.assertFalse(mask[0, :3].any())
        self.assertFalse(mask[0, :, :3].any())
        self.assertEqual(torch.nonzero(mask[0, 4], as_tuple=False).flatten().tolist(), [3, 4])

    def test_zero_windows_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_ultra_attention_mask(
                lengths=torch.tensor([2], dtype=torch.long),
                max_seq_len=2,
                local_window=0,
                global_window=0,
                truncation_length=0,
                layer_idx=0,
                num_full_layers=0,
            )


if __name__ == "__main__":
    unittest.main()
