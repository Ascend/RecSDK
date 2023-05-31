#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from mx_rec.util.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from mx_rec.util.tf_version_adapter import npu_ops, hccl_ops
from mx_rec.saver.patch import patch_for_saver
from mx_rec.graph.patch import patch_for_dataset
from mx_rec.optimizers.base import patch_for_optimizer


patch_for_saver()
patch_for_dataset()
patch_for_optimizer()
__version__ = "5.0.RC2"


def version():
    return __version__
