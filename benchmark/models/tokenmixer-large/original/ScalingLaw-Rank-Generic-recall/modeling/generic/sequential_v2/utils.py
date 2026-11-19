from typing import Tuple

import torch


def handle_padded_qk(B: int, cached_k, cached_q, delta_x_offsets: Tuple[torch.Tensor, torch.Tensor], k, n: int, q):
    """
    处理填充的Q和K, 以适应变长序列的注意力机制。

    在自注意力(Self-Attention)机制中, 为了处理不同长度的序列, 需要对序列进行填充(Padding)。
    此函数用于将缓存中的Q(Query)和K(Key)与实际的Q和K进行对齐, 以便正确计算注意力分数。

    参数:
    - B: 批次大小(Batch size)。
    - cached_k: 缓存中的Key张量。
    - cached_q: 缓存中的Query张量。
    - delta_x_offsets: 用于计算偏移量的张量。
    - k: 实际的Key张量。
    - n: 序列长度。
    - q: 实际的Query张量。

    返回:
    - padded_k: 填充后的Key张量。
    - padded_q: 填充后的Query张量。
    """
    padded_q, padded_k = cached_q, cached_k
    # 将偏移量张量展平，并计算实际的索引位置
    flattened_offsets = delta_x_offsets[1] + torch.arange(start=0, end=B * n, step=n,
                                                          device=delta_x_offsets[1].device,
                                                          dtype=delta_x_offsets[1].dtype)
    # 使用索引复制q到padded_q中，然后恢复其原始形状
    padded_q = padded_q.view(B * n, -1).index_copy_(
        dim=0, index=flattened_offsets, source=q,
    ).view(B, n, -1)
    # 使用索引复制k到padded_k中，然后恢复其原始形状
    padded_k = padded_k.view(B * n, -1).index_copy_(
        dim=0, index=flattened_offsets, source=k,
    ).view(B, n, -1)
    return padded_k, padded_q


def dense_to_jagged_1d(dense, x_offsets, total_l: int = 0):
    jagged_length = int(x_offsets[-1])

    if total_l > 0:
        if total_l >= jagged_length:
            jagged_length = total_l
        else:
            raise ValueError("total_l %s is less than jagged_length %s" % (total_l, jagged_length))

    num_sequences = x_offsets.shape[0] - 1

    jagged_size = [jagged_length] + list(dense.shape[2:])

    jagged_data = torch.full(jagged_size, 0, dtype=dense.dtype, device=dense.device)

    for i in range(num_sequences):
        start = x_offsets[i]
        end = x_offsets[i + 1]
        seq_length = end - start
        jagged_data[start:end] = dense[i, :seq_length]

    return jagged_data


def jagged_1d_to_padded_dense(values, offsets, max_length: int, padding_value: float = 0):
    num_sequences = offsets.shape[0] - 1

    dense_size = [num_sequences, max_length] + list(values.shape[1:])

    padded_data = torch.full(dense_size, padding_value, dtype=values.dtype, device=values.device)

    for i in range(num_sequences):
        start = offsets[i]
        end = offsets[i + 1]
        seq_length = end - start
        padded_data[i, :seq_length] = values[start:end]

    return padded_data


def get_last_n_embeddings(
        lengths: torch.Tensor,
        encoded_embeddings: torch.Tensor,
        num_rerank: int
) -> torch.Tensor:
    B, N, D = encoded_embeddings.size()

    # Ensure num_rerank is not greater than the actual length of each sequence
    num_rerank = min(num_rerank, N)

    # Initialize an empty tensor to store the results
    last_n_embeddings = torch.zeros((B, num_rerank, D), device=encoded_embeddings.device)
    lengths_cpu = lengths.cpu().numpy()
    for i in range(B):
        # Calculate the start index for each sequence
        start_idx = lengths_cpu[i]

        # Calculate the end index (exclusive)
        end_idx = start_idx + num_rerank

        # Directly slice the embeddings between start_idx and end_idx
        last_n_embeddings[i, :, :] = encoded_embeddings[i, start_idx:end_idx, :]

    return last_n_embeddings

def get_current_embeddings(
        lengths: torch.Tensor,
        encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (
            (lengths - 1)
            + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    )
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)

