import torch


def _check_int_valid(num: int):
    if not isinstance(num, int):
        return False
    if num <= 0:
        return False
    return True


def create_target_mask(num_target: int, target_group_size: int) -> torch.Tensor:
    row_indices = torch.arange(num_target).view(-1, 1)
    col_indices = torch.arange(num_target).view(1, -1)

    block_row = row_indices // target_group_size
    block_col = col_indices // target_group_size

    mask = (block_row == block_col).int()
    tril = torch.tril(torch.ones(num_target, num_target), diagonal=0).int()
    return tril & mask


def create_causal_mask(
    seqlen_q: int,
    seqlen_k: int = None,
    num_context: int = None,
    num_target: int = None,
    target_group_size: int = None,
) -> torch.Tensor:
    if seqlen_k is None:
        seqlen_k = seqlen_q
    # causal mask
    mask = torch.tril(torch.ones(seqlen_q, seqlen_k), diagonal=(seqlen_k - seqlen_q))
    # context mask
    if _check_int_valid(num_context):
        num_target = 0 if num_target is None else num_target
        mask[:num_context, : seqlen_k - num_target] = 1
    # target mask
    if _check_int_valid(target_group_size) and _check_int_valid(num_target):
        mask[-num_target:, -num_target:] = create_target_mask(num_target, target_group_size)
    return mask


def create_arbitrary_mask(
    seqlen_q: int, seqlen_k: int = None, groups: int = 2, data_type: torch.dtype = torch.float16
) -> (torch.Tensor, torch.Tensor):
    if seqlen_k is None:
        seqlen_k = seqlen_q
    mask = torch.zeros(seqlen_q, seqlen_k, dtype=data_type)
    arbitrary_func = torch.zeros(seqlen_q, 2 * groups, dtype=torch.int32)
    for i in range(seqlen_q):
        arbitrary_func[i, :] = torch.randint(seqlen_k, size=(2 * groups,)).sort().values
        if arbitrary_func[i, :].all() == 0:
            arbitrary_func[i, 1] = 1
        for g in range(groups):
            st, ed = arbitrary_func[i, g * 2 : g * 2 + 2].tolist()
            mask[i, st:ed] = 1
    return mask, arbitrary_func


def create_causal_arbitrary_mask(
    seqlen_q: int, seqlen_k: int = None, groups: int = 2, data_type: torch.dtype = torch.float16
) -> (torch.Tensor, torch.Tensor):
    if seqlen_k is None:
        seqlen_k = seqlen_q
    mask = torch.zeros(seqlen_q, seqlen_k, dtype=data_type)
    arbitrary_func = torch.zeros(seqlen_q, 2 * groups, dtype=torch.int32)
    deltaqk = max(1, seqlen_k - seqlen_q + 1)
    for i in range(seqlen_q):
        arbitrary_func[i, -1] = deltaqk + i
        mask[i, : deltaqk + i] = 1
    return mask, arbitrary_func


def create_batch_arbitrary_mask(
    batch_size: int,
    head_num: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seq_offset_q: torch.Tensor,
    seq_offset_k: torch.Tensor,
    groups: int = 2,
    data_type: torch.dtype = torch.float16,
) -> (torch.Tensor, torch.Tensor):
    seqlens_q = seq_offset_q[1:] - seq_offset_q[:-1]
    seqlens_k = seq_offset_k[1:] - seq_offset_k[:-1]
    mask = torch.zeros(batch_size, head_num, max_seqlen_q, max_seqlen_k, dtype=data_type)
    arbitrary_func = torch.zeros(batch_size, 1, max_seqlen_q, groups * 2)
    for i, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k)):
        _mask, _arbitrary_func = create_causal_arbitrary_mask(seqlen_q, seqlen_k, groups)
        mask[i, :, :seqlen_q, :seqlen_k] = _mask
        arbitrary_func[i, :, :seqlen_q, :] = _arbitrary_func
    return mask, arbitrary_func


def _build_sparse_info(
    mask: torch.Tensor,
    seq_offset_q: torch.Tensor,
    seq_offset_k: torch.Tensor,
    block_q: int,
    block_kv: int,
    k2q: bool,
):
    """通用 sparse_info 构建。k2q=True 外层遍历 K 块、内层遍历 Q 块;False 反之。"""
    batch_size, _, max_seqlen_q, max_seqlen_k = mask.shape
    max_blklen = (max_seqlen_k + block_kv - 1) // block_kv if k2q else (max_seqlen_q + block_q - 1) // block_q
    seqlens_q = seq_offset_q[1:] - seq_offset_q[:-1]
    seqlens_k = seq_offset_k[1:] - seq_offset_k[:-1]

    mask_cnt = torch.zeros(batch_size, 1, max_blklen, dtype=torch.int32)
    mask_offset, mask_idx = [0], []
    full_cnt = torch.zeros(batch_size, 1, max_blklen, dtype=torch.int32)
    full_offset, full_idx = [0], []

    for b, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k)):
        n_q_blk = (seqlen_q + block_q - 1) // block_q
        n_k_blk = (seqlen_k + block_kv - 1) // block_kv
        n_outer, n_inner = (n_k_blk, n_q_blk) if k2q else (n_q_blk, n_k_blk)

        for outer_blk in range(n_outer):
            blk_mask_cnt = blk_full_cnt = 0
            for inner_blk in range(n_inner):
                if k2q:
                    q, k = inner_blk * block_q, outer_blk * block_kv
                else:
                    q, k = outer_blk * block_q, inner_blk * block_kv
                q_end = min(q + block_q, seqlen_q)
                k_end = min(k + block_kv, seqlen_k)
                tensor = mask[b, 0, q:q_end, k:k_end]
                max_val, min_val = tensor.max(), tensor.min()
                if min_val == 1:
                    blk_full_cnt += 1
                    full_idx.append(inner_blk)
                elif max_val == 1:
                    blk_mask_cnt += 1
                    mask_idx.append(inner_blk)
            mask_cnt[b, 0, outer_blk] = blk_mask_cnt
            full_cnt[b, 0, outer_blk] = blk_full_cnt
            mask_offset.append(mask_offset[-1] + blk_mask_cnt)
            full_offset.append(full_offset[-1] + blk_full_cnt)
        # padding 块补齐 offset,保证每 batch 恰好 max_blklen 项
        mask_offset += [mask_offset[-1]] * (max_blklen - n_outer)
        full_offset += [full_offset[-1]] * (max_blklen - n_outer)

    return (
        mask_cnt,
        torch.tensor(mask_offset, dtype=torch.int32),
        torch.tensor(mask_idx, dtype=torch.int32),
        full_cnt,
        torch.tensor(full_offset, dtype=torch.int32),
        torch.tensor(full_idx, dtype=torch.int32),
    )


def create_q2k_sparse_info(
    mask: torch.Tensor, seq_offset_q: torch.Tensor, seq_offset_k: torch.Tensor, block_q: int, block_kv: int
):
    return _build_sparse_info(mask, seq_offset_q, seq_offset_k, block_q, block_kv, k2q=False)


def create_k2q_sparse_info(
    mask: torch.Tensor, seq_offset_q: torch.Tensor, seq_offset_k: torch.Tensor, block_q: int, block_kv: int
):
    return _build_sparse_info(mask, seq_offset_q, seq_offset_k, block_q, block_kv, k2q=True)
