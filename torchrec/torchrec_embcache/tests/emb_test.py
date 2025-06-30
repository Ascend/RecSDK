#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys

import embedding_cache
import torch
from acc_test.util import setup_logging
from torchrec_embcache.saver import Saver

setup_logging(0)


# 假设我们有两个 embedding table
emb_configs = [
    embedding_cache.EmbConfig(table_name="table1", emb_dim=8, optim_num=1, cache_size=10000, weight_init_min=0.0,
                              weight_init_max=1.0),
    embedding_cache.EmbConfig(table_name="table2", emb_dim=4, optim_num=1, cache_size=5000, weight_init_min=-1.0,
                              weight_init_max=0.0),
]

# 假设每个表都是单值特征
multihot_sizes = [1, 1]

# 初始化 EmbcacheManager
manager = embedding_cache.EmbcacheManager(emb_configs)

# 构造一个 batch 的 key，这里假设 batch_size=4，每个样本有两个特征
# table1: [100, 200, 300, 400]
# table2: [1000, 2000, 3000, 4000]
batch_keys = torch.tensor([100, 1000, 200, 2000, 300, 3000, 400, 4000], dtype=torch.int64)
jagged_offs = [0, 2, 4, 6, 8]

# 1. 异步获取 SwapInfo
swap_info_task = manager.compute_swap_info_async(batch_keys, jagged_offs)
swap_info = swap_info_task.get()

logging.debug("SwapInfo:")
logging.debug("  Swapout Keys: %s", swap_info.swapout_keys)
logging.debug("  Swapout Offsets: %s", swap_info.swapout_offs)
logging.debug("  Swapin Keys: %s", swap_info.swapin_keys)
logging.debug("  Swapin Offsets: %s", swap_info.swapin_offs)
logging.debug("  Batch Offsets: %s", swap_info.batch_offs)

# 2. 异步进行 Embedding Lookup
swapin_tensor_task = manager.embedding_lookup_async(swap_info.swapin_keys)
swapin_tensor = swapin_tensor_task.get()

logging.debug("\nSwapin Tensor Details:")
logging.debug("  Swapin Embeddings: %s", swapin_tensor.swapin_embs)
logging.debug("  Swapin Optimizers: %s", swapin_tensor.swapin_optims)
logging.debug("  Jagged Offsets: %s", swapin_tensor.jagged_offs)

# 3. 进行 Embedding Update（假设我们有一些梯度）
# 假设 swapout_embs 和 swapout_optims 是从外部获取的，例如通过 optimizer.step()
# 这里为了演示，我们生成一些 tensor
swapout_embs = torch.cat((torch.arange(0.1, 0.1 + 2 * emb_configs[0].emb_dim * 0.1, 0.1, dtype=torch.float32),
                          torch.arange(-0.1, -0.1 - 2 * emb_configs[1].emb_dim * 0.1, -0.1,
                                       dtype=torch.float32))).reshape(1, -1)
logging.debug("  Swapout Embeddings: %s", swapout_embs)
swapout_optims = torch.cat((torch.arange(1, 1 + 2 * emb_configs[0].emb_dim * 1, 1, dtype=torch.float32),
                            torch.arange(-1, -1 - 2 * emb_configs[1].emb_dim * 1, -1, dtype=torch.float32))).reshape(1,
                                                                                                                     -1)
logging.debug("  Swapout Optimizers: %s", swapout_optims)
manager.embedding_update(
    torch.tensor([[100, 200], [300, 400]], dtype=torch.int64),
    swapout_embs,
    swapout_optims
)
logging.debug("\nEmbedding updated.")

# 4. Retrieve the updated embeddings to verify correctness
updated_tensor_task = manager.embedding_lookup_async(
    torch.tensor([[100, 200], [300, 400]], dtype=torch.int64)
)
updated_tensor = updated_tensor_task.get()

logging.debug("\nUpdated Tensor Details:")
logging.debug("  Updated Embeddings: %s", updated_tensor.swapin_embs)
logging.debug("  Updated Optimizers: %s", updated_tensor.swapin_optims)
logging.debug("  Jagged Offsets: %s", updated_tensor.jagged_offs)

manager.save("save_dir", 0)

from emb_read import emb_read, compare_embedding_dicts
dt1 = emb_read("save_dir")

manager.load("save_dir", 0)
manager.save("save_dir2", 0)
dt2 = emb_read("save_dir2")
if not compare_embedding_dicts(dt1, dt2):
    raise ValueError("Embedding dictionaries do not match after save and load.")