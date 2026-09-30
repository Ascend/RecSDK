import pytest
import torch
import ops_rec

from utils import get_block_q_kv
from utils.mask import create_k2q_sparse_info, create_q2k_sparse_info
from utils.mask import create_batch_arbitrary_mask

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)


def ops_rec_create_sparse(func_tensor, q_len, kv_len, q_block_size, kv_block_size, mode):
    if mode == "fwd":
        _out = ops_rec.ascendc.attention.create_q2k_block_sparse_from_func(
            func_tensor.npu(), q_len, kv_len, q_block_size, kv_block_size
        )
    else:
        _out = ops_rec.ascendc.attention.create_k2q_block_sparse_from_func(
            func_tensor.npu(), q_len, kv_len, q_block_size, kv_block_size
        )

    return ops_rec.ascendc.attention.compact_no_sync(*_out)


def _run_test_case(batch_size, max_seqlen_qk, head_num, head_dim, groups):
    seq_lens_q = torch.randint(1, max_seqlen_qk + 1, (batch_size,), dtype=torch.int32)
    seq_lens_k = torch.randint(1, max_seqlen_qk + 1, (batch_size,), dtype=torch.int32)
    seq_lens_q = torch.where(seq_lens_k < seq_lens_q, seq_lens_k, seq_lens_q)

    seq_offset_q = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_q, axis=0))).numpy()
    seq_offset_k = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_k, axis=0))).numpy()

    mask, arbitrary_func = create_batch_arbitrary_mask(
        batch_size, head_num, max_seqlen_qk, max_seqlen_qk, seq_offset_q, seq_offset_k, groups, torch.float16
    )

    BLOCK_Q, BLOCK_KV = get_block_q_kv(head_dim, head_dim, "fwd")
    q2k_sparse_info = create_q2k_sparse_info(mask, seq_offset_q, seq_offset_k, BLOCK_Q, BLOCK_KV)
    q2k_sparse_info_npu = ops_rec_create_sparse(arbitrary_func, max_seqlen_qk, max_seqlen_qk, BLOCK_Q, BLOCK_KV, "fwd")
    for gt, pred in zip(q2k_sparse_info, q2k_sparse_info_npu):
        assert torch.equal(gt, pred.cpu())

    BLOCK_Q, BLOCK_KV = get_block_q_kv(head_dim, head_dim, "bwd")
    k2q_sparse_info = create_k2q_sparse_info(mask, seq_offset_q, seq_offset_k, BLOCK_Q, BLOCK_KV)
    k2q_sparse_info_npu = ops_rec_create_sparse(arbitrary_func, max_seqlen_qk, max_seqlen_qk, BLOCK_Q, BLOCK_KV, "bwd")
    for gt, pred in zip(k2q_sparse_info, k2q_sparse_info_npu):
        assert torch.equal(gt, pred.cpu())


@pytest.mark.parametrize("batch_size, head_num, seq_lens", [(32, 8, (1024, 1024))])
@pytest.mark.parametrize("head_dims", [(64, 64), (128, 128)])
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("seed", [123])
def test_user_case_1(
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    groups,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    _run_test_case(batch_size, max_seqlen_q, head_num, head_dim_qk, groups)


@pytest.mark.parametrize("batch_size, head_num, seq_lens", [(128, 4, (8186, 8186))])
@pytest.mark.parametrize("head_dims", [(128, 128)])
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("seed", [123])
def test_user_case_2(
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    groups,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    _run_test_case(batch_size, max_seqlen_q, head_num, head_dim_qk, groups)


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("head_num", [4])
@pytest.mark.parametrize("head_dims", [(32, 32), (256, 256)])
@pytest.mark.parametrize("seq_lens", [(128, 128), (1234, 1234)])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("seed", [123])
def test_has_mask(
    batch_size,
    head_num,
    head_dims,
    seq_lens,
    groups,
    seed,
):
    head_dim_qk, head_dim_v = head_dims
    max_seqlen_q, max_seqlen_k = seq_lens
    _run_test_case(batch_size, max_seqlen_q, head_num, head_dim_qk, groups)
