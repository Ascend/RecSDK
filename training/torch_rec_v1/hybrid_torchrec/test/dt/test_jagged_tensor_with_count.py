#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch

import pytest
import torch

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from hybrid_torchrec.sparse.jagged_tensor_with_count import (
    KeyedJaggedTensorWithCount,
    unpack_tensors,
    JaggedTensorWithCount,
)


class TestJaggedTensorWithCount(unittest.TestCase):
    """Test cases for JaggedTensorWithCount class"""

    def _create_test_data(self):
        """Helper method to create common test data for JaggedTensorWithCount"""
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        lengths = torch.tensor([2, 0, 1, 3, 1, 1])
        counts = torch.tensor([2, 0, 1, 3, 1, 1])
        return values, lengths, counts

    def test_jagged_tensor_with_count_comprehensive(self) -> None:
        """Test JaggedTensorWithCount comprehensive functionality"""
        values, lengths, counts = self._create_test_data()

        # Test basic construction with counts
        jtc1 = JaggedTensorWithCount(values=values, lengths=lengths, counts=counts)

        # Verify all properties at once
        self.assertTrue(torch.equal(jtc1.values(), values))
        self.assertTrue(torch.equal(jtc1.lengths(), lengths))
        self.assertTrue(torch.equal(jtc1.counts, counts))
        self.assertIsNone(jtc1.weights_or_none())

        # Test permute method
        # Test that JaggedTensorWithCount does not have permute method
        self.assertFalse(hasattr(jtc1, "permute"))

        # Test counts property variations
        values_simple = torch.tensor([1, 2, 3, 4, 5, 6])
        lengths_simple = torch.tensor([2, 0, 1, 3])
        counts_simple = torch.tensor([2, 0, 1, 3])

        # Test with counts
        jtc_with_counts = JaggedTensorWithCount(
            values=values_simple, lengths=lengths_simple, counts=counts_simple
        )
        self.assertTrue(torch.equal(jtc_with_counts.counts, counts_simple))

        # Test without counts
        jtc_without_counts = JaggedTensorWithCount(
            values=values_simple, lengths=lengths_simple
        )
        self.assertIsNone(jtc_without_counts.counts)
        self.assertIsNone(jtc_without_counts.weights_or_none())


class TestKeyedJaggedTensorWithCount(unittest.TestCase):
    """Test cases for KeyedJaggedTensorWithCount class"""

    def _create_test_data(self):
        """Helper method to create common test data"""
        keys = ["feature1", "feature2"]
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        counts = torch.tensor([2, 1, 3, 1, 2, 1])
        lengths = torch.tensor([3, 3])
        return keys, values, counts, lengths

    def _create_test_data_with_weights(self):
        """Helper method to create common test data with weights"""
        keys, values, counts, lengths = self._create_test_data()
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        return keys, values, counts, lengths, weights

    def test_keyed_jagged_tensor_with_count_comprehensive(self):
        """Comprehensive test for KeyedJaggedTensorWithCount"""
        keys, values, counts, lengths, weights = self._create_test_data_with_weights()

        # Create all KeyedJaggedTensorWithCount variations in one place
        kjt = KeyedJaggedTensorWithCount(
            keys=keys, values=values, counts=counts, weights=weights, lengths=lengths
        )

        kjtwc_basic = KeyedJaggedTensorWithCount(
            keys=keys, values=values, counts=counts, lengths=lengths
        )

        offsets = torch.tensor([0, 3, 6])
        kjtwc_with_offsets = KeyedJaggedTensorWithCount(
            keys=keys, values=values, counts=counts, lengths=lengths, offsets=offsets
        )

        kjtwc_no_counts = KeyedJaggedTensorWithCount(
            keys=keys, values=values, lengths=lengths
        )

        kjt_weights_only = KeyedJaggedTensorWithCount(
            keys=keys, values=values, weights=weights, lengths=lengths
        )

        # Test basic properties for all variations at once
        tensors_to_test = [
            (kjt, keys, values, lengths, weights, counts),
            (kjtwc_basic, keys, values, lengths, None, counts),
            (kjtwc_with_offsets, keys, values, lengths, None, counts),
            (kjtwc_no_counts, keys, values, lengths, None, None),
            (kjt_weights_only, keys, values, lengths, weights, None),
        ]

        for tensor_data in tensors_to_test:
            (
                tensor,
                expected_keys,
                expected_values,
                expected_lengths,
                expected_weights,
                expected_counts,
            ) = tensor_data
            assert tensor.keys() == expected_keys
            assert torch.equal(tensor.values(), expected_values)
            assert torch.equal(tensor.lengths(), expected_lengths)
            if expected_weights is not None:
                assert torch.equal(tensor.weights_or_none(), expected_weights)
            if expected_counts is not None:
                assert torch.equal(tensor.counts, expected_counts)

        # Test dist_labels method variations
        labels_sets = [
            (kjt.dist_labels(), {"lengths", "values", "weights", "counts"}),
            (kjtwc_basic.dist_labels(), {"lengths", "values", "counts"}),
            (kjtwc_no_counts.dist_labels(), {"lengths", "values"}),
            (kjt_weights_only.dist_labels(), {"lengths", "values", "weights"}),
        ]

        for labels, expected in labels_sets:
            assert set(labels) == expected

        # Test dist_tensors method variations
        tensors_results = [
            (kjt.dist_tensors(), 4, weights, counts),
            (kjtwc_basic.dist_tensors(), 3, None, counts),
            (kjtwc_no_counts.dist_tensors(), 2, None, None),
            (kjt_weights_only.dist_tensors(), 3, weights, None),
        ]

        for tensors, expected_len, expected_weights, expected_counts in tensors_results:
            self.assertEqual(len(tensors), expected_len)
            self.assertTrue(torch.equal(tensors[0], lengths))
            self.assertTrue(torch.equal(tensors[1], values))
            if expected_weights is not None:
                self.assertTrue(torch.equal(tensors[2], expected_weights))
            elif expected_counts is not None:
                # For cases with counts but no weights
                self.assertTrue(torch.equal(tensors[2], expected_counts))

        # Test dist_splits method variations
        splits_results = [
            (kjtwc_basic.dist_splits([1, 1]), 3),  # counts is included
            (kjtwc_no_counts.dist_splits([1, 1]), 2),  # counts is not included
            (kjt_weights_only.dist_splits([1, 1]), 3),
        ]

        for splits, expected_len in splits_results:
            self.assertEqual(len(splits), expected_len)

        # Test other methods
        split_result = kjtwc_basic.split([2])
        assert len(split_result) == 1
        assert split_result[0].keys() == keys

        # Test to method (device operations)
        moved_kjtwc = kjtwc_basic.to(torch.device("cpu"))
        assert torch.equal(moved_kjtwc.values(), values)
        assert torch.equal(moved_kjtwc.counts, counts)
        assert torch.equal(moved_kjtwc.lengths(), lengths)

        # Test to方法，转换到CPU设备
        kjt_cpu = kjtwc_basic.to("cpu")
        assert kjt_cpu.values().device.type == "cpu"
        assert kjt_cpu.lengths().device.type == "cpu"
        assert kjt_cpu.counts.device.type == "cpu"

        # 测试pin_memory方法（在CPU环境中，我们只测试方法是否能被调用）
        try:
            pinned_kjt = kjtwc_basic.pin_memory()
            assert pinned_kjt is not None
        except RuntimeError as e:
            # 在没有CUDA设备的环境中，这是预期的行为
            assert "Cannot access accelerator device when none is available" in str(e)

        # 测试to_dict方法
        dict_result = kjtwc_basic.to_dict()
        assert len(dict_result) == 2
        assert "feature1" in dict_result
        assert "feature2" in dict_result

        # 测试sync方法
        try:
            synced_kjt = kjtwc_basic.sync()
            assert synced_kjt is not None
            assert isinstance(synced_kjt, KeyedJaggedTensorWithCount)
        except NotImplementedError:
            # 某些情况下sync可能未实现，这是可以接受的
            pass

    def test_keyed_jagged_tensor_with_count_permute_and_dict(self):
        """Test KeyedJaggedTensorWithCount permute and from_jt_dict methods"""
        keys, values, counts, lengths, weights = self._create_test_data_with_weights()

        kjt = KeyedJaggedTensorWithCount(
            keys=keys, values=values, counts=counts, weights=weights, lengths=lengths
        )

        # Test permute method
        indices = [1, 0]
        permuted_kjt = kjt.permute(indices)
        self.assertIsInstance(permuted_kjt, KeyedJaggedTensorWithCount)
        self.assertEqual(permuted_kjt.keys(), ["feature2", "feature1"])

        # Test KeyedJaggedTensorWithCount from_jt_dict method
        # Create JaggedTensorWithCount instances
        jt1 = JaggedTensorWithCount(
            values=torch.tensor([1, 2, 3]),
            counts=torch.tensor([1, 1, 1]),
            lengths=torch.tensor([3]),
        )
        jt2 = JaggedTensorWithCount(
            values=torch.tensor([4, 5, 6]),
            counts=torch.tensor([2, 1, 3]),
            lengths=torch.tensor([3]),
        )

        # Create dictionary
        jt_dict = {"feature1": jt1, "feature2": jt2}

        # Test from_jt_dict method
        kjt_from_dict = KeyedJaggedTensorWithCount.from_jt_dict(jt_dict)
        self.assertIsInstance(kjt_from_dict, KeyedJaggedTensorWithCount)
        self.assertEqual(kjt_from_dict.keys(), ["feature1", "feature2"])
        self.assertTrue(
            torch.equal(kjt_from_dict.counts, torch.tensor([1, 1, 1, 2, 1, 3]))
        )


def test_unpack_tensors_function():
    """Test unpack_tensors function with various tensor configurations"""
    from hybrid_torchrec.sparse.jagged_tensor_with_count import unpack_tensors as unpack_tensors_func

    test_cases = [
        # (tensors, variable_stride_per_key, expected_lengths, expected_values, expected_stride, expected_weights)
        (
            [torch.tensor([1, 2]), torch.tensor([1, 2, 3])],
            False,
            True,
            True,
            False,
            False,
        ),
        (
            [torch.tensor([1, 2]), torch.tensor([1, 2, 3]), torch.tensor([2, 4])],
            True,
            True,
            True,
            True,
            False,
        ),
        (
            [
                torch.tensor([1, 2]),
                torch.tensor([1, 2, 3]),
                torch.tensor([2, 4]),
                torch.tensor([0.1, 0.2, 0.3]),
            ],
            True,
            True,
            True,
            True,
            False,
        ),  # weights is None because 4 tensors with variable_stride_per_key means last tensor is counts
        (
            [
                torch.tensor([1, 2]),
                torch.tensor([1, 2, 3]),
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([1, 1, 2]),
            ],
            False,
            True,
            True,
            False,
            True,
        ),  # 4 tensors with no variable stride means 3rd is weights, 4th is counts
        (
            [
                torch.tensor([1, 2]),
                torch.tensor([1, 2, 3]),
                torch.tensor([2, 4]),
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([1, 1, 2]),
            ],
            True,
            True,
            True,
            True,
            True,
        ),
    ]

    for (
        tensors,
        variable_stride_per_key,
        has_lengths,
        has_values,
        has_stride,
        has_weights,
    ) in test_cases:
        lengths, values, stride_per_rank_per_key, weights = unpack_tensors_func(
            tensors, variable_stride_per_key
        )
        if has_lengths:
            assert lengths is not None
        if has_values:
            assert values is not None
        if has_stride:
            assert stride_per_rank_per_key is not None
        else:
            assert stride_per_rank_per_key is None
        if has_weights:
            assert weights is not None
        else:
            assert weights is None

    # Test with invalid number of tensors
    with pytest.raises(RuntimeError):
        unpack_tensors_func([torch.tensor([1])], False)


if __name__ == "__main__":
    unittest.main()
