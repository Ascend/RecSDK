#!/usr/bin/env python3
# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import unittest
from unittest import mock

from mx_rec.graph.slicers import NoGradSubgraphSlicer
from mx_rec.graph.hooks import LookupSubgraphSlicerHook, OrphanLookupKeySlicerHook


class MockLookupSubgraphSlicer(NoGradSubgraphSlicer):
    def __init__(self, op_types) -> None:
        super().__init__()

    def summarize(self) -> None:
        pass

    def slice(self) -> None:
        pass


class MockOrphanLookupKeySlicer(NoGradSubgraphSlicer):
    def __init__(self) -> None:
        super().__init__()

    def summarize(self) -> None:
        pass

    def slice(self) -> None:
        pass


class TestLookupSubgraphSlicerHook(unittest.TestCase):
    @mock.patch.multiple(
        "mx_rec.graph.hooks",
        LookupSubgraphSlicer=mock.MagicMock(return_value=MockLookupSubgraphSlicer(["xxx"])),
    )
    def test_ok(self):
        hook = LookupSubgraphSlicerHook(["xxx"])
        hook.begin()
        self.assertIsNotNone(hook)


class TestOrphanLookupKeySlicerHook(unittest.TestCase):
    @mock.patch.multiple(
        "mx_rec.graph.hooks",
        OrphanLookupKeySlicer=mock.MagicMock(return_value=MockOrphanLookupKeySlicer()),
    )
    def test_ok(self):
        hook = OrphanLookupKeySlicerHook()
        hook.begin()
        self.assertIsNotNone(hook)
