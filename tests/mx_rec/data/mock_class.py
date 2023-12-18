#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.


class MockEosOpsLib:
    """
    mock librec.eos_dataset
    """

    def __init__(self, variant_tensor):
        def _mock_eos_dataset_fn(**kwargs):
            return variant_tensor

        self.eos_dataset = _mock_eos_dataset_fn
