#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
from typing import Dict, Any, Union, List


import numpy as np


class SeqStats:
    """序列长度统计类，用于计算序列长度的最大值和最小值"""

    @staticmethod
    def compute_seq_stats(
        seq_offset_q: Union[np.ndarray, List],
        seq_offset_k: Union[np.ndarray, List],
        max_seqlen_q: int,
        max_seqlen_k: int
    ) -> Dict[str, Any]:
        """从 seq_offset 计算序列长度统计信息

        Args:
            seq_offset_q: Q 序列的偏移量数组
            seq_offset_k: K 序列的偏移量数组
            max_seqlen_q: Q 序列的最大长度
            max_seqlen_k: K 序列的最大长度

        Returns:
            包含统计信息和概率分布的字典
        """
        # 从 seq_offset 恢复 seq_lens
        seq_lens_q = np.diff(seq_offset_q)
        seq_lens_k = np.diff(seq_offset_k)

        return SeqStats._calc_stats(seq_lens_q, seq_lens_k, max_seqlen_q, max_seqlen_k)

    @staticmethod
    def compute_seq_lens(
        seq_offset_q: Union[np.ndarray, List],
        seq_offset_k: Union[np.ndarray, List]
    ) -> tuple:
        """从 seq_offset 恢复 seq_lens

        Args:
            seq_offset_q: Q 序列的偏移量数组
            seq_offset_k: K 序列的偏移量数组

        Returns:
            (seq_lens_q, seq_lens_k): 恢复后的序列长度数组
        """
        seq_lens_q = np.diff(seq_offset_q)
        seq_lens_k = np.diff(seq_offset_k)
        return seq_lens_q, seq_lens_k

    @staticmethod
    def _calc_stats(
        seq_lens_q: np.ndarray,
        seq_lens_k: np.ndarray,
        max_seqlen_q: int,
        max_seqlen_k: int
    ) -> Dict[str, Any]:
        """计算序列长度统计信息

        Args:
            seq_lens_q: Q 序列长度数组
            seq_lens_k: K 序列长度数组
            max_seqlen_q: Q 序列的最大长度
            max_seqlen_k: K 序列的最大长度

        Returns:
            包含统计信息和概率分布的字典
        """
        # 计算基本统计值
        stats = {
            "seq_lens_q_mean": float(np.mean(seq_lens_q)),
            "seq_lens_q_max": int(np.max(seq_lens_q)),
            "seq_lens_q_min": int(np.min(seq_lens_q)),
            "seq_lens_k_mean": float(np.mean(seq_lens_k)),
            "seq_lens_k_max": int(np.max(seq_lens_k)),
            "seq_lens_k_min": int(np.min(seq_lens_k)),
        }

        return stats