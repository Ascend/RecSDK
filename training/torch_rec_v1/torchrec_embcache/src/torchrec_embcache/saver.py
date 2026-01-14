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
import shutil

import torch.distributed as dist
import torch.nn

from torchrec_embcache.distributed.embedding_bag import EmbCacheShardedEmbeddingBagCollection
from torchrec_embcache.distributed.embedding import EmbCacheShardedEmbeddingCollection
from torchrec_embcache.utils import check_path, safe_makedirs


SAVE_PATH_MAX_LEN = 1024
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_MAX_RECURSIVE_TIMES = 500
_MAX_LOOP_TIMES = 500
_OPEN_FILE_MODE = 0o640
_OPEN_FILE_FLAGS = os.O_CREAT | os.O_WRONLY | os.O_TRUNC


class Saver:
    def __init__(self, rank: int = None):
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = torch.distributed.get_world_size()
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
        self.world_size: int = world_size
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
            raise ValueError(f"expect a timestamp directory but empty in path")
        latest_dir = max(d.name for d in dirs)
        return os.path.join(os.path.realpath(path), latest_dir)

    @staticmethod
    def _get_format_path():
        return datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FORMAT)

    def save(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:
        check_path(path)
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"param `module` must an instance of torch.nn.Module, but got:{type(module)}")

        if not dist.is_initialized():
            raise ValueError("when save, the status of torch.distributed.is_initialized() must be True, but got False.")

        path = os.path.realpath(path)

        self.cache_module.clear()
        self._find_all_embed_cache_instance(module)
        self._check_emb_cache_instance_len()
        safe_makedirs(path)
        if self.rank == 0:
            meta_path = os.path.join(path, "meta.ini")
            with os.fdopen(os.open(meta_path, _OPEN_FILE_FLAGS, _OPEN_FILE_MODE), "w") as f:
                f.write(f"incremental={incremental}\n")
        logging.info("In save scene, path:%s, cache_module info:%s", path, self.cache_module)
        for mod in self.cache_module:
            logging.info("In save scene, embcache_mgr info:%s", mod.embcache_mgr)
            codegen = mod.get_batched_embedding_kernels()[0][0]
            momentum_list = [momentum.detach().to("cpu") for momentum in codegen.get_momentum()] 
            mod.embcache_mgr.embedding_to_host(codegen.weights_dev.detach().to("cpu"), momentum_list)
            mod.embcache_mgr.save(path, self.rank, incremental=incremental)
            dist.barrier()
            if self.rank == 0:
                mod.embcache_mgr.merge_files(path, self.world_size)
            dist.barrier()
            table_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            for table_dir in table_dirs:
                rank_path = os.path.join(path, table_dir, f"rank{self.rank}")
                shutil.rmtree(rank_path)

    def load(self, module: torch.nn.Module, path: str, incremental=False) -> None:
        check_path(path, need_exist=True, is_dir=True)
        self.cache_module.clear()
        self._find_all_embed_cache_instance(module)
        self._check_emb_cache_instance_len()
        path = os.path.realpath(path)
        check_path(path)
        for mod in self.cache_module:
            mod.embcache_mgr.load(path, self.rank, self.world_size, incremental)

    def _find_all_embed_cache_instance(self, module: EmbCacheShardedEmbeddingBagCollection, this_recur_step: int = 0):
        if this_recur_step >= _MAX_RECURSIVE_TIMES:
            raise RuntimeError(f"Recursion depth not greater than {_MAX_RECURSIVE_TIMES}")
        for ind, (_, child) in enumerate(module.named_children()):
            if ind >= _MAX_LOOP_TIMES:
                raise RuntimeError(f"Len of module children should not be greater than {_MAX_LOOP_TIMES}")
            if (isinstance(child, EmbCacheShardedEmbeddingBagCollection)
                    or isinstance(child, EmbCacheShardedEmbeddingCollection)):
                self.cache_module.append(child)
            self._find_all_embed_cache_instance(child, this_recur_step + 1)

    def _check_emb_cache_instance_len(self):
        if len(self.cache_module) == 0:
            raise ValueError("param `module` must has at least one child module which "
                             "type is EmbCacheShardedEmbeddingBagCollection or EmbCacheShardedEmbeddingBagCollection.")
