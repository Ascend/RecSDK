import sys
from unittest.mock import MagicMock
sys.modules['torch_npu'] = MagicMock

import pytest
import torch
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import bucketize_kjt_before_all2all
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestBucketizeKJTBeforeAll2All:
    @staticmethod
    def create_kjt(num_features: int = 2,
                   num_lengths: int = 10,
                   max_lengths: int = 10):
        keys = [f'feature_{i}' for i in range(num_features)]
        lengths = torch.randint(0, max_lengths, (num_features * num_lengths,))
        values = torch.randint(0, 10, size=(lengths.sum().tolist(),))
        kjt = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        return kjt

    @pytest.mark.parametrize("world_size", [1, 2, 3])
    def test_basic_bucketize(self, world_size):
        kjt = self.create_kjt()
        length = len(kjt.lengths()) // len(kjt.keys())
        block_sizes = [kjt.lengths()[i * length: i * length + length].sum() for i in range(len(kjt.keys()))]
        block_sizes = torch.tensor(block_sizes)
        bucketize_kjt_before_all2all(kjt, world_size, block_sizes)
