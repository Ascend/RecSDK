#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import abc

from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.core.emb.dynamic_sparse_embedding import HBMDynamicSparseEmbedding
from mx_rec.core.emb.mergeable_sparse_embedding import MergeableSparseEmbedding
from mx_rec.core.emb.sparse_embedding import (
    ExternalStorageSparseEmbedding,
    HBMSparseEmbedding,
)
from mx_rec.util.singleton import singleton


class BaseSparseEmbeddingFactory(metaclass=abc.ABCMeta):
    """
    Base class for sparse embedding table creation.
    """

    @abc.abstractmethod
    def create_embedding(self, config: dict) -> BaseSparseEmbedding:
        """Method used for the practical creation process.

        Args:
            config: Params for embedding table creation.

        Returns: Sparse embedding table.
        """
        pass


@singleton
class HBMDynamicSparseEmbeddingFactory(BaseSparseEmbeddingFactory):
    """
    Embedding table factory using pure HBM storage with dynamic expansion.
    """

    def create_embedding(self, config: dict) -> HBMDynamicSparseEmbedding:
        return HBMDynamicSparseEmbedding(config)


@singleton
class HBMSparseEmbeddingFactory(BaseSparseEmbeddingFactory):
    """
    Embedding table factory using pure HBM storage without dynamic expansion.
    """

    def create_embedding(self, config: dict) -> HBMSparseEmbedding:
        return HBMSparseEmbedding(config)


@singleton
class ExternalStorageSparseEmbeddingFactory(BaseSparseEmbeddingFactory):
    """
    Embedding table factory using DDR and SSD storage, which regrad HBM as high speed cache.
    """

    def create_embedding(self, config: dict) -> ExternalStorageSparseEmbedding:
        return ExternalStorageSparseEmbedding(config)
