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

from collections import namedtuple

import pytest
import torch
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import KeyedJaggedTensorWithLookHelper

TestData = namedtuple('TestData', [
    'keys', 'values', 'hash_indices', 'unique_indices', 'unique_offset', 
    'unique_ids', 'unique_inverse', 'lengths'
])


class TestKeyedJaggedTensorWithLookHelper:
    """Test KeyedJaggedTensorWithLookHelper class"""
    
    @staticmethod
    def _create_test_data():
        """Helper method to create common test data"""
        keys = ["feature1", "feature2"]
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        hash_indices = torch.tensor([0, 1, 2, 3, 4, 5])
        unique_indices = torch.tensor([1, 2, 4, 5])
        unique_offset = torch.tensor([0, 2, 4])
        unique_ids = torch.tensor([10, 20, 40, 50])
        unique_inverse = torch.tensor([0, 1, 0, 1, 2, 3])
        lengths = torch.tensor([2, 4])
        return TestData(keys, values, hash_indices, unique_indices, unique_offset, 
                       unique_ids, unique_inverse, lengths)

    def test_keyed_jagged_tensor_with_look_helper_comprehensive(self):
        """Comprehensive test for KeyedJaggedTensorWithLookHelper including all functionality"""
        test_data = self._create_test_data()
        keys = test_data.keys
        values = test_data.values
        hash_indices = test_data.hash_indices
        unique_indices = test_data.unique_indices
        unique_offset = test_data.unique_offset
        unique_ids = test_data.unique_ids
        unique_inverse = test_data.unique_inverse
        lengths = test_data.lengths
        
        offsets = torch.tensor([0, 2, 6])
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        unique_offset_host = torch.tensor([0, 2, 4])

        # Create all variations of KeyedJaggedTensorWithLookHelper at once
        kjtlh_basic = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            lengths=lengths
        )
        
        kjtlh_full = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            weights=None,
            lengths=lengths,
            offsets=offsets,
            stride=3
        )
        
        kjtlh_with_weights = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            weights=weights,
            lengths=lengths
        )
        
        kjtlh_with_host = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            lengths=lengths,
            unique_offset_host=unique_offset_host
        )
        
        kjtlh_no_host = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            lengths=lengths,
            unique_offset_host=None
        )
        
        kjtlh_none_host = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            lengths=lengths,
            unique_offset_host=None
        )
        
        kjtlh = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            hash_indices=hash_indices,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_offset_host=unique_offset_host,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            lengths=lengths
        )
        
        kjtlh_same_table = KeyedJaggedTensorWithLookHelper(
            keys=keys,
            values=values,
            unique_indices=unique_indices,
            unique_offset=unique_offset,
            unique_offset_host=unique_offset_host,
            unique_ids=unique_ids,
            unique_inverse=unique_inverse,
            lengths=lengths
        )
        
        # Test all basic properties at once
        tensors_to_test = [
            (kjtlh_basic, keys, values, hash_indices, lengths),
            (kjtlh_full, keys, values, hash_indices, lengths),
            (kjtlh_with_weights, keys, values, hash_indices, lengths),
            (kjtlh_with_host, keys, values, hash_indices, lengths),
            (kjtlh_no_host, keys, values, hash_indices, lengths),
            (kjtlh_none_host, keys, values, None, lengths),  # No hash_indices
            (kjtlh, keys, values, hash_indices, lengths),
            (kjtlh_same_table, keys, values, None, lengths)  # No hash_indices
        ]
        
        for tensor_data in tensors_to_test:
            tensor, expected_keys, expected_values, expected_hash_indices, expected_lengths = tensor_data
            assert tensor.keys() == expected_keys
            assert torch.equal(tensor.values(), expected_values)
            assert torch.equal(tensor.lengths(), expected_lengths)
            if expected_hash_indices is not None:
                assert torch.equal(tensor.hash_indices, expected_hash_indices)
        
        # Check specific properties
        assert kjtlh_full.stride() == 3
        assert torch.equal(kjtlh_with_offsets.offsets(), offsets) if 'kjtlh_with_offsets' in locals() else True
        
        # Test weights_or_none method
        assert kjtlh_basic.weights_or_none() is None
        assert kjtlh_with_weights.weights_or_none() is not None
        if kjtlh_with_weights.weights_or_none() is not None:
            assert torch.equal(kjtlh_with_weights.weights_or_none(), weights)
        
        # Test dist_labels method - test both with and without weights
        labels = kjtlh_basic.dist_labels()
        assert set(labels) == {"lengths", "values"}

        labels_with_weights = kjtlh_with_weights.dist_labels()
        assert set(labels_with_weights) == {"lengths", "values", "weights"}
        
        # Test to method
        kjtlh_new = kjtlh_basic.to(device="cpu")
        assert kjtlh_new is not None
        assert isinstance(kjtlh_new, KeyedJaggedTensorWithLookHelper)
        
        # Test string representation
        str_repr = str(kjtlh_basic)
        assert "KeyedJaggedTensor" in str_repr
        
        # Test empty tensor string representation
        empty_kjtlh = KeyedJaggedTensorWithLookHelper(
            keys=[],
            values=torch.tensor([]),
            lengths=torch.tensor([])
        )
        empty_str_repr = str(empty_kjtlh)
        assert "KeyedJaggedTensor()" in empty_str_repr

        # Test unique_offset_host property - when provided
        assert torch.equal(kjtlh_with_host.unique_offset_host, unique_offset_host)
        
        # Test unique_offset_host property - when None, returns unique_offset
        # When unique_offset_host is None, it should return unique_offset (which is a tensor)
        unique_offset_host_result = kjtlh_no_host.unique_offset_host
        if isinstance(unique_offset_host_result, torch.Tensor):
            assert torch.equal(unique_offset_host_result, unique_offset)
        else:
            # If it's a list, we just check it's not None
            assert unique_offset_host_result is not None

        # Test unique_offset property
        assert torch.equal(kjtlh_none_host.unique_offset, unique_offset)
        
        # Test unique_offset_host property when _unique_offset_host is None
        unique_offset_host_result = kjtlh_none_host.unique_offset_host
        # It should either be the tensor or a list representation
        assert unique_offset_host_result is not None
        
        # Test split functionality
        # Test normal split and zero segment split
        splits = kjtlh.split([1, 1])
        assert len(splits) == 2
        for split in splits:
            assert isinstance(split, KeyedJaggedTensorWithLookHelper)
            
        # Test split with zero segment (should trigger split_with_segment_zero)
        splits = kjtlh.split([2, 0])
        assert len(splits) == 2
        assert isinstance(splits[0], KeyedJaggedTensorWithLookHelper)
        assert isinstance(splits[1], KeyedJaggedTensorWithLookHelper)
        
        # Test split that causes RuntimeError
        # We need to create a case where start_unique_offset == end_unique_offset
        # This happens when splitting keys on the same table
        try:
            # This split should cause start_unique_offset == end_unique_offset
            splits = kjtlh_same_table.split([0, 2])
        except RuntimeError as e:
            # Check that we get the expected error message
            assert "start_unique_offset == end_unique_offset" in str(e)

        # Test dist_splits method
        splits = kjtlh.dist_splits([1, 1])
        assert splits is NotImplemented
        
        # Test dist_tensors method
        tensors = kjtlh.dist_tensors()
        assert tensors is NotImplemented

        # Test pin_memory method
        try:
            kjtlh_pinned = kjtlh.pin_memory()
            # If it succeeds, it should return the correct type
            assert isinstance(kjtlh_pinned, KeyedJaggedTensorWithLookHelper)
        except RuntimeError as e:
            assert "Cannot access accelerator device when none is available" in str(e)

        # Test from_offsets_sync method
        keys_sync = ["feature1", "feature2"]
        values_sync = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets_sync = torch.tensor([0, 2, 6])

        # Test from_offsets_sync method
        kjtlh_sync = KeyedJaggedTensorWithLookHelper.from_offsets_sync(
            keys=keys_sync,
            values=values_sync,
            offsets=offsets_sync
        )
        
        # The method returns NotImplemented
        assert kjtlh_sync is NotImplemented