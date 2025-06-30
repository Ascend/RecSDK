#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random

import embedding_cache
import torch
from acc_test.util import setup_logging


setup_logging(0)


class Gen:
    def __init__(self, max_key):
        self.max_key = max_key

    def gen_keys(self, batch_size):
        return [random.randint(0, self.max_key - 1) for _ in range(batch_size)]


def swap_manager_test(batch_size, cache_size, test_num):
    gen = Gen(cache_size * 10)
    emb_bag_config = embedding_cache.EmbConfig()
    emb_bag_config.table_name = "test_table"
    emb_bag_config.emb_dim = 128  # Example dimension
    emb_bag_config.optim_num = 2  # Example optim number
    emb_bag_config.cache_size = cache_size

    manager = embedding_cache.EmbcacheManager([emb_bag_config])  # Assuming single-hot

    cache = [0] * cache_size
    key2off = {}

    for t in range(test_num):
        logging.debug("\n===================== Testing batch %d =====================", t)
        keys = gen.gen_keys(batch_size)
        keys_bak = keys.copy()
        key_set = set(keys)

        batch_keys_tensor = torch.tensor(keys, dtype=torch.int64)
        jagged_offs_tensor = [0, batch_size]

        # Get SwapInfo
        swap_info_future = manager.compute_swap_info_async(batch_keys_tensor, jagged_offs_tensor)
        swap_info = swap_info_future.get()

        swapout_keys = swap_info.swapout_keys[0]
        swapout_offs = swap_info.swapout_offs
        swapin_keys = swap_info.swapin_keys[0]
        swapin_offs = swap_info.swapin_offs
        batch_offs = swap_info.batch_offs
        logging.debug("swapout size: %d", len(swapout_keys))
        logging.debug("swapin size: %d", len(swapin_keys))
        logging.debug()

        # 1. key 和 off size 相同
        if len(swapout_keys) != len(swapout_offs):
            raise ValueError("Mismatch in size of swapout_keys and swapout_offs")
        if len(swapin_keys) != len(swapin_offs):
            raise ValueError("Mismatch in size of swapin_keys and swapin_offs")

        # 2. 执行 swapout，验证 swapoutKeys 和 swapoutOffs 都在 cache 中，
        # 且 swapoutKeys 不在 keys 中，且 off 都在范围内
        for key, off in zip(swapout_keys, swapout_offs):
            if off >= cache_size:
                raise ValueError(f"swapout offset {off} out of range")
            if cache[off] != key:
                raise ValueError(f"swapout key {key} not found at offset {off} in cache")
            if key not in key2off:
                raise ValueError(f"swapout key {key} not found in key2off")
            if key in key_set:
                raise ValueError(f"swapout key {key} unexpectedly found in key_set")

            cache[off] = 0
            del key2off[key]

        # 3. 执行 swapin，验证 swapinKeys 都在 keys 中，
        # 且 swapinKeys 和 swapoutOffs 都不在 cache 中，且 off 都在范围内
        for key, off in zip(swapin_keys, swapin_offs):
            if off >= cache_size:
                raise ValueError(f"swapin offset {off} out of range")
            if key not in key_set:
                raise ValueError(f"swapin key {key} not found in key_set")
            if key in key2off:
                raise ValueError(f"swapin key {key} unexpectedly found in key2off")

            cache[off] = key
            key2off[key] = off

        # 4. 验证 keys 都在 cache 中，且 key 都转化成 off
        for key, off in zip(keys_bak, batch_offs):
            cache[off] = key
            key2off[key] = off
            if key not in key2off:
                raise ValueError(f"Key {key} not found in key2off after batch processing")
            if cache[off] != key:
                raise ValueError(f"Key {key} not found at offset {off} in cache after batch processing")
            if key2off[key] != off:
                raise ValueError(f"Offset mismatch for key {key}: expected {off}, got {key2off[key]}")

        logging.debug("Test %d passed!", t)


if __name__ == "__main__":
    batch_size = 0
    cache_size = 0
    test_num = 100
    swap_manager_test(batch_size, cache_size, test_num)

    batch_size = 1
    cache_size = 1
    test_num = 100
    swap_manager_test(batch_size, cache_size, test_num)

    batch_size = 10000
    cache_size = 10000
    test_num = 100
    swap_manager_test(batch_size, cache_size, test_num)
