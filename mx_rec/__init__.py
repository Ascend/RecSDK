#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from .util.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION
from .util.tf_version_adapter import npu_ops, hccl_ops
from .saver.patch import patch_for_saver
from .graph.patch import patch_for_dataset


patch_for_saver()
patch_for_dataset()
