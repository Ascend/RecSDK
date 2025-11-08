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
  
from unittest.mock import patch

import torch
import pytest

from torchrec.sparse.jagged_tensor import JaggedTensor
from hybrid_torchrec.sparse.extended_jagged_tensor import (
    ExtendedJaggedTensor,
    KeyedExtendedJaggedTensor,
)


class TestExtendedJaggedTensor:
    """Test ExtendedJaggedTensor class"""
    
    @staticmethod
    def _create_test_data():
        """Helper method to create common test data"""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        extra = torch.tensor([1, 1, 2, 2, 3, 3])
        lengths = torch.tensor([2, 0, 3, 1])
        return values, extra, lengths

    def test_extended_jagged_tensor_comprehensive(self):
        """Comprehensive test for ExtendedJaggedTensor"""
        values, extra, lengths = self._create_test_data()
        offsets = torch.tensor([0, 2, 2, 5, 6])
        
        # Test all initialization variations
        ejt_basic = ExtendedJaggedTensor(values=values, extra=extra, lengths=lengths)
        ejt_no_weights = ExtendedJaggedTensor(values=values, extra=extra, weights=None, lengths=lengths)
        ejt_with_offsets = ExtendedJaggedTensor(values=values, extra=extra, lengths=lengths, offsets=offsets)
        
        # Verify basic properties for all variations
        assert torch.equal(ejt_basic.values(), values)
        assert torch.equal(ejt_basic.extra, extra)
        assert torch.equal(ejt_basic.lengths(), lengths)
        
        assert torch.equal(ejt_no_weights.values(), values)
        assert torch.equal(ejt_no_weights.lengths(), lengths)
        # weights() method may not exist or behave differently, we only verify the object can be created normally
        
        assert torch.equal(ejt_with_offsets.values(), values)
        assert torch.equal(ejt_with_offsets.extra, extra)
        assert torch.equal(ejt_with_offsets.lengths(), lengths)
        assert torch.equal(ejt_with_offsets.offsets(), offsets)
        
        # Test field validation
        # Test with mismatched field tensor size
        try:
            # Create an ExtendedJaggedTensor with a mismatched field tensor size
            ejt = ExtendedJaggedTensor(
                values=values,
                extra=extra,
                lengths=lengths,
                _fields="_extra"
            )
            # Manually set a field attribute with wrong size
            setattr(ejt, "_extra", torch.tensor([1, 2]))  # Wrong size
            # If we get here, the exception was not raised as expected
            assert False, "Expected TypeError was not raised"
        except TypeError:
            # This is expected
            pass


class TestKeyedExtendedJaggedTensor:
    """Test KeyedExtendedJaggedTensor class"""
    
    @staticmethod
    def _create_test_data():
        """Helper method to create common test data"""
        keys = ["feature1", "feature2"]
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        extra = torch.tensor([1, 1, 2, 2, 3, 3])
        lengths = torch.tensor([2, 4])
        return keys, values, extra, lengths

    def test_keyed_extended_jagged_tensor_comprehensive(self):
        """Comprehensive test for KeyedExtendedJaggedTensor"""
        keys, values, extra, lengths = self._create_test_data()
        offsets = torch.tensor([0, 2, 6])
        
        # Test all initialization variations
        kejt_basic = KeyedExtendedJaggedTensor(
            keys=keys, values=values, extra=extra, lengths=lengths
        )
        kejt_no_weights = KeyedExtendedJaggedTensor(
            keys=keys, values=values, extra=extra, weights=None, lengths=lengths
        )
        kejt_with_offsets = KeyedExtendedJaggedTensor(
            keys=keys, values=values, extra=extra, lengths=lengths, offsets=offsets
        )
        
        # Test all basic properties at once
        basic_props = [
            (kejt_basic, keys, values, extra, lengths),
            (kejt_no_weights, keys, values, extra, lengths),
            (kejt_with_offsets, keys, values, extra, lengths, offsets)
        ]
        
        for props in basic_props:
            obj = props[0]
            assert obj.keys() == props[1]
            assert torch.equal(obj.values(), props[2])
            assert torch.equal(obj.extra, props[3])
            assert torch.equal(obj.lengths(), props[4])
            if len(props) > 5:  # Has offsets
                assert torch.equal(obj.offsets(), props[5])
        
        # Test weights_or_none method
        assert kejt_no_weights.weights_or_none() is None
        
        # Test split_extend method variations in one place
        segments = [1, 1]  # Split into 2 segments
        split_result = kejt_basic.split_extend(segments, KeyedExtendedJaggedTensor)
        assert len(split_result) == 2
        assert split_result[0].keys() == ["feature1"]
        assert split_result[1].keys() == ["feature2"]
        
        
        # Test split with zero segment (should create empty tensor)
        split_result = kejt_basic.split_extend([0, 2], KeyedExtendedJaggedTensor)
        assert len(split_result) == 2
        assert split_result[0].keys() == []  # Empty segment
        assert split_result[1].keys() == keys  # All keys in second segment
        
        # Verify empty segment has correct properties
        empty_segment = split_result[0]
        assert empty_segment.values().numel() == 0
        if empty_segment.extra is not None:
            assert empty_segment.extra.numel() == 0

        # Test permute_extend method variations
        indices = [1, 0]  # Reorder features
        permuted_kejt = kejt_basic.permute_extend(
            indices, None, KeyedExtendedJaggedTensor
        )
        assert permuted_kejt.keys() == ["feature2", "feature1"]
        
        
        # Test split with segments longer than keys (should raise error)
        with pytest.raises(IndexError, match="list index out of range"):
            kejt_basic.split_extend([3], KeyedExtendedJaggedTensor)
        
        # Test with another set of data for additional scenarios
        keys_3 = ["feature1", "feature2", "feature3"]
        values_3 = torch.tensor([1, 2, 3, 4, 5, 6])
        extra_3 = torch.tensor([1, 1, 2, 2, 3, 3])
        lengths_3 = torch.tensor([2, 2, 2])
        
        kejt = KeyedExtendedJaggedTensor(
            keys=keys_3, values=values_3, extra=extra_3, lengths=lengths_3
        )
        
        # Test split with normal segment
        split_result = kejt.split_extend([1, 2], KeyedExtendedJaggedTensor)
        assert len(split_result) == 2
        assert split_result[0].keys() == ["feature1"]
        assert split_result[1].keys() == ["feature2", "feature3"]
        
        # Verify extra field slicing happened
        assert split_result[0].extra is not None
        assert split_result[1].extra is not None
        
        # Create mock JT dict
        jt_dict = {}
        values_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
        lengths_list = [torch.tensor([2, 1]), torch.tensor([2, 2])]

        for i, key in enumerate(["feature1", "feature2"]):
            jt = JaggedTensor(
                values=values_list[i],
                lengths=lengths_list[i]
            )
            jt_dict[key] = jt

        # Test _construct_from_jt_dict with proper arguments and correct parameter order
        self.construct_from_jt_dict(kejt_basic, jt_dict)
        
        # Mock conditions to cover lines 271-273
        with patch('hybrid_torchrec.sparse.extended_jagged_tensor.is_non_strict_exporting',
                   return_value=True):
            self.construct_from_jt_dict(kejt, jt_dict)

        # Test regular path (lines 274-290)
        self.construct_from_jt_dict(kejt, jt_dict)
        
        # Test permute with indices tensor for extra field processing
        indices = [2, 0, 1]  # Reorder features
        indices_tensor = torch.tensor(indices, dtype=torch.int32)
        permuted_kejt = kejt.permute_extend(
            indices, indices_tensor, KeyedExtendedJaggedTensor
        )
        assert permuted_kejt.keys() == ["feature3", "feature1", "feature2"]
        # Check that extra fields are also properly permuted
        assert permuted_kejt.extra is not None
        
        # Test KeyedExtendedJaggedTensor permute with various conditions using mock
        keys_2 = ["feature1", "feature2"]
        values_2 = torch.tensor([1, 2, 3, 4, 5, 6])
        extra_2 = torch.tensor([1, 1, 2, 2, 3, 3])
        lengths_2 = torch.tensor([2, 4])
        
        kejt_2 = KeyedExtendedJaggedTensor(
            keys=keys_2, values=values_2, extra=extra_2, lengths=lengths_2
        )
        
        # Mock is_non_strict_exporting to cover lines 265-267
        with patch('hybrid_torchrec.sparse.extended_jagged_tensor.is_non_strict_exporting', 
                   return_value=True):
            self.permute_extend(kejt_2)

        # Test permute with torchdynamo compilation path
        self.permute_extend(kejt_2)

        # Mock conditions to cover lines 306-329
        with patch('hybrid_torchrec.sparse.extended_jagged_tensor.is_torchdynamo_compiling', 
                   return_value=True), \
             patch('torch.jit.is_scripting', return_value=False):
            self.permute_extend(kejt_2)
            
        # Test KeyedExtendedJaggedTensor with variable stride configuration
        # Using smaller and more reasonable stride_per_key_per_rank configuration to avoid integer wraparound issues
        stride_per_key_per_rank = [[1], [1]]
        
        # Create KeyedExtendedJaggedTensor with variable stride
        kejt_variable = KeyedExtendedJaggedTensor(
            keys=keys_2,
            values=values_2,
            extra=extra_2,
            lengths=lengths_2,
            stride_per_key_per_rank=stride_per_key_per_rank
        )
        
        # Test that variable stride is properly set
        assert kejt_variable.variable_stride_per_key() == True
        assert kejt_variable.stride_per_key_per_rank() == stride_per_key_per_rank
        
        # Test permute with variable stride
        indices = [1, 0]
        indices_tensor = torch.tensor(indices, dtype=torch.int32)
        permuted_kejt = kejt_variable.permute_extend(
            indices, indices_tensor, KeyedExtendedJaggedTensor
        )
        assert permuted_kejt.keys() == ["feature2", "feature1"]
        assert permuted_kejt.variable_stride_per_key() == True

    @staticmethod
    def construct_from_jt_dict(kejt_basic, jt_dict):
        constructed_kejt = KeyedExtendedJaggedTensor._construct_from_jt_dict(
            kejt_basic, jt_dict, KeyedExtendedJaggedTensor, lambda x: x.values()
        )
        assert constructed_kejt.keys() == ["feature1", "feature2"]

    @staticmethod
    def permute_extend(kejt_2):
        indices = [1, 0]
        indices_tensor = torch.tensor(indices, dtype=torch.int32)
        
        # This should trigger the torchdynamo compilation path
        permuted_kejt = kejt_2.permute_extend(
            indices, indices_tensor, KeyedExtendedJaggedTensor
        )
        assert permuted_kejt.keys() == ["feature2", "feature1"]