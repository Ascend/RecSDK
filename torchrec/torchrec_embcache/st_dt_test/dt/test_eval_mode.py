#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu
from torchrec_embcache import EmbcacheManager, EmbConfig, AdmitAndEvictConfig, InitializerType as InitializerTypeCpp

torch.npu.set_device(0)


def test_eval_mode():
    new_configs = [
        EmbConfig(
            table_name="test1",
            initializer_type=getattr(InitializerTypeCpp, "LINEAR"),
            emb_dim=4,
            optim_num=1,
            cache_size=1000,
            weight_init_min=0.0,
            weight_init_max=1.0,
            weight_init_mean=0.5,
            weight_init_stddev=0.5,
            admit_and_evict_config=AdmitAndEvictConfig(),
            initializer_random_pool_size=1000,
            seed=0,
        )
    ]

    embcache_manager = EmbcacheManager(emb_configs=new_configs, need_accumulate_offset=False)
    embcache_manager.set_eval_mode(new_key_return_0=True)
    batch_keys = torch.tensor([100, 1000, 200, 2000, 300, 3000, 400, 4000], dtype=torch.int64)
    jagged_offs = [0, 7]
    swap_info = embcache_manager.compute_swap_info_async(batch_keys, jagged_offs).get()
    assert all([off == -1000 for off in swap_info.batch_offs]), "batch_offs should be -1000"


if __name__ == "__main__":
    test_eval_mode()