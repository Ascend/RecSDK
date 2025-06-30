#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

from torchrec_embcache.distributed.embedding_bag import EmbCacheShardedEmbeddingBagCollection
from torchrec_embcache.distributed.embedding import EmbCacheShardedEmbeddingCollection


class Saver:
    def __init__(self, rank=None):
        if rank is None:
            logging.warning("rank id is None, will set rank id as 0.")
        self.rank = rank
        self.cache_module = []

    def save(self, module, path):
        self.cache_module.clear()
        self.find_all_embed_cache_instance(module)
        logging.info("In save scene, cache_module info:%s", self.cache_module)
        for mod in self.cache_module:
            logging.info("In save scene, embcache_mgr info:%s", mod.embcache_mgr)
            codegen = mod.get_batched_embedding_kernels()[0][0]
            momentum_list = [momentum.detach().to("cpu") for momentum in codegen.get_momentum()] 
            mod.embcache_mgr.embedding_to_host(codegen.weights_dev.detach().to("cpu"), momentum_list)
            mod.embcache_mgr.save(path, self.rank)

    def find_all_embed_cache_instance(self, module):
        for _, child in module.named_children():
            if (isinstance(child, EmbCacheShardedEmbeddingBagCollection)
                    or isinstance(child, EmbCacheShardedEmbeddingCollection)):
                self.cache_module.append(child)
            self.find_all_embed_cache_instance(child)

    def load(self, module, path):
        self.cache_module.clear()
        self.find_all_embed_cache_instance(module)
        for mod in self.cache_module:
            mod.embcache_mgr.load(path, self.rank)
