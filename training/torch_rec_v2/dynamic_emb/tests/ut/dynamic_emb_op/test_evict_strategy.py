#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
import torch

import dynamic_emb_extensions as demb


class EvictStrategyOptions:
    def __init__(self):
        self.evict_strategy_lru = demb.EvictStrategy.kLru
        self.evict_strategy_lfu = demb.EvictStrategy.kLfu
        self.evict_strategy_epoch_lru = demb.EvictStrategy.kEpochLru
        self.evict_strategy_epoch_lfu = demb.EvictStrategy.kEpochLfu
        self.evict_strategy_customized = demb.EvictStrategy.kCustomized


@pytest.fixture
def evict_strategy_options():
    return EvictStrategyOptions()


def test_evict_strategy(evict_strategy_options: EvictStrategyOptions):
    assert evict_strategy_options.evict_strategy_lru == demb.EvictStrategy.kLru
    assert evict_strategy_options.evict_strategy_lfu == demb.EvictStrategy.kLfu
    assert evict_strategy_options.evict_strategy_epoch_lru == demb.EvictStrategy.kEpochLru
    assert evict_strategy_options.evict_strategy_epoch_lfu == demb.EvictStrategy.kEpochLfu
    assert evict_strategy_options.evict_strategy_customized == demb.EvictStrategy.kCustomized
    