#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch.distributed as dist
import torch.nn

from torchrec_embcache.distributed.embedding_bag import EmbCacheShardedEmbeddingBagCollection
from torchrec_embcache.distributed.embedding import EmbCacheShardedEmbeddingCollection
from torchrec_embcache.utils import check_path


SAVE_PATH_MAX_LEN = 1024
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_SAVE_PATH_MIN_LEN = 1
_DIR_MODE = 0o750


class Saver:
    def __init__(self, rank: int = None):
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
                logging.warning("Param rank id is None and distributed model has been initialized,"
                                " get rank by dist.get_rank() is:%d", rank)
            else:
                raise ValueError("param `rank` must not be None when torch.distributed.is_initialized() is False.")
        else:
            if not (isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0):
                raise ValueError("param rank must be an integer and need greater or equal than 0.")
            if dist.is_initialized():
                world_size = torch.distributed.get_world_size()
                if rank >= world_size:
                    raise ValueError(f"param `rank` must less than torch distribution world_size:{world_size},"
                                     f" but got rank {rank}")
        self.rank: int = rank
        self.cache_module = []

    @staticmethod
    def is_timestamp_format(dir_name: str) -> bool:
        try:
            datetime.strptime(dir_name, TIMESTAMP_FORMAT)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_latest_load_path(path):
        path = Path(path)
        dirs = [d for d in path.iterdir() if d.is_dir() and Saver.is_timestamp_format(d.name)]
        if not dirs:
            raise ValueError(f"expect a timestamp directory but empty in path:{path}")
        latest_dir = max(d.name for d in dirs)
        return os.path.join(os.path.realpath(path), latest_dir)

    @staticmethod
    def _get_format_path():
        return datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT)

    def save(self, module: torch.nn.Module, path: str) -> None:
        check_path(path)
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"param `module` must an instance of torch.nn.Module, but got:{type(module)}")

        if not dist.is_initialized():
            raise ValueError("when save, the status of torch.distributed.is_initialized() must be True, but got False.")

        path = os.path.realpath(path)
        dist.barrier()
        timestamp_data = int(Saver._get_format_path()) if self.rank == 0 else 0
        timestamp_tensor = torch.tensor([timestamp_data], device="npu")
        dist.broadcast(timestamp_tensor, src=0)
        timestamp_str = str(timestamp_tensor[0].item())
        logging.info("rank:%d, after broadcast, get current timestamp: %s", self.rank, timestamp_str)
        path = os.path.join(path, timestamp_str)

        self.cache_module.clear()
        self._find_all_embed_cache_instance(module)
        self._check_emb_cache_instance_len()
        if not os.path.exists(path):
            os.makedirs(path, _DIR_MODE, exist_ok=True)
        logging.info("In save scene, path:%s, cache_module info:%s", path, self.cache_module)
        for mod in self.cache_module:
            logging.info("In save scene, embcache_mgr info:%s", mod.embcache_mgr)
            codegen = mod.get_batched_embedding_kernels()[0][0]
            momentum_list = [momentum.detach().to("cpu") for momentum in codegen.get_momentum()] 
            mod.embcache_mgr.embedding_to_host(codegen.weights_dev.detach().to("cpu"), momentum_list)
            mod.embcache_mgr.save(path, self.rank)

    def load(self, module: torch.nn.Module, path: str) -> None:
        check_path(path, need_exist=True, is_dir=True)
        self.cache_module.clear()
        self._find_all_embed_cache_instance(module)
        self._check_emb_cache_instance_len()
        path = os.path.realpath(path)
        path = self.get_latest_load_path(path)
        check_path(path)
        for mod in self.cache_module:
            mod.embcache_mgr.load(path, self.rank)

    def _find_all_embed_cache_instance(self, module):
        for _, child in module.named_children():
            if (isinstance(child, EmbCacheShardedEmbeddingBagCollection)
                    or isinstance(child, EmbCacheShardedEmbeddingCollection)):
                self.cache_module.append(child)
            self._find_all_embed_cache_instance(child)

    def _check_emb_cache_instance_len(self):
        if len(self.cache_module) == 0:
            raise ValueError("param `module` must has at least one child module which "
                             "type is EmbCacheShardedEmbeddingBagCollection or EmbCacheShardedEmbeddingBagCollection.")
