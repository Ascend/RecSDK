import torch

def batch_gather_embeddings(
        rowwise_indices: torch.Tensor,
        embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        rowwise_indices: (B, N) x int, where each entry is in [0, X).
        embeddings: (B, X, D,) x float.

    Returns:
        (B, N, D,) x float, embeddings corresponding to rowwise_indices.
    """
    _, N = rowwise_indices.size()
    B, X, D = embeddings.size()
    flattened_indices = (
            rowwise_indices
            + torch.arange(
        start=0, end=B, step=1, dtype=rowwise_indices.dtype, device=rowwise_indices.device
    ).unsqueeze(1).expand(-1, N) * X
    )
    return embeddings.view(-1, D)[flattened_indices, :].reshape(rowwise_indices.size() + (D,))
