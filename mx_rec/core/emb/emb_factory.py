#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import abc

from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.core.emb.dynamic_sparse_embedding import HBMDynamicSparseEmbedding
from mx_rec.core.emb.sparse_embedding import HBMSparseEmbedding, ExternalStorageSparseEmbedding


class BaseSparseEmbeddingFactory(metaclass=abc.ABCMeta):
    """
    创建Embedding的工厂基类.
    """

    @abc.abstractmethod
    def create_embedding(self, config: dict) -> BaseSparseEmbedding:
        """
        创建embedding类.

        Args:
            config: 创建embedding所需的参数字典.

        Returns: embedding类
        """
        pass


class HBMDynamicSparseEmbeddingFactory(BaseSparseEmbeddingFactory):
    """
    HBMDynamicSparseEmbedding工厂.
    """

    def create_embedding(self, config: dict) -> HBMDynamicSparseEmbedding:
        return HBMDynamicSparseEmbedding(config)


class HBMSparseEmbeddingFactory(BaseSparseEmbeddingFactory):
    """
    HBMSparseEmbedding工厂.
    """

    def create_embedding(self, config: dict) -> HBMSparseEmbedding:
        return HBMSparseEmbedding(config)


class ExternalStorageSparseEmbeddingFactory(BaseSparseEmbeddingFactory):
    """
    ExternalStorageSparseEmbedding工厂.
    """

    def create_embedding(self, config: dict) -> ExternalStorageSparseEmbedding:
        return ExternalStorageSparseEmbedding(config)
