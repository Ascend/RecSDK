import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.sequential_v2.transformers import RMSNormNPU
from modeling.generic.utils.constants import Const
from modeling.model_registry import ModelRegistry


NPU_ENABLE = os.environ.get("NPU_FLAG", "True") != "False"
TP_ALL_GATHER_INTO_TENSOR = os.environ.get("TP_ALL_GATHER_INTO_TENSOR", "1") != "0"
if NPU_ENABLE:
    try:
        import torch_npu  # type: ignore
    except Exception:
        torch_npu = None
else:
    torch_npu = None


def _get_npu_op(name: str):
    if torch_npu is not None:
        op = getattr(torch_npu, name, None)
        if op is not None:
            return op
    if hasattr(torch.ops, "npu"):
        return getattr(torch.ops.npu, name, None)
    return None


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _split_range(total: int, rank: int, size: int) -> Tuple[int, int]:
    base = total // size
    extra = total % size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def _pad_dim(x: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
    if dim < 0:
        dim += x.dim()
    current = x.size(dim)
    if current == target_size:
        return x
    if current > target_size:
        raise ValueError("target_size must be greater than or equal to current dim size")
    padding = [0] * (2 * x.dim())
    padding[2 * (x.dim() - dim - 1) + 1] = target_size - current
    return F.pad(x, tuple(padding), mode="constant", value=0.0)


def _all_gather_padded_dim(
    x: torch.Tensor,
    group,
    group_size: int,
    dim: int,
    sizes: Tuple[int, ...],
    max_size: int,
) -> torch.Tensor:
    padded = _pad_dim(x, dim, max_size).contiguous()
    if TP_ALL_GATHER_INTO_TENSOR and hasattr(dist, "all_gather_into_tensor"):
        send = padded.movedim(dim, 0).contiguous()
        recv_shape = (int(group_size) * int(max_size),) + tuple(send.shape[1:])
        recv = torch.empty(recv_shape, dtype=send.dtype, device=send.device)
        dist.all_gather_into_tensor(recv, send, group=group)
        return _assemble_gathered_buffer(recv, dim, sizes, max_size)

    gathered = [torch.empty_like(padded) for _ in range(int(group_size))]
    dist.all_gather(gathered, padded, group=group)
    gathered = [part.narrow(dim, 0, sizes[idx]) for idx, part in enumerate(gathered)]
    return torch.cat(gathered, dim=dim)


def _assemble_gathered_buffer(
    recv: torch.Tensor,
    dim: int,
    sizes: Tuple[int, ...],
    max_size: int,
) -> torch.Tensor:
    if all(size == int(max_size) for size in sizes):
        if dim == 0:
            return recv
        return recv.movedim(0, dim).contiguous()
    chunks = torch.split(recv, int(max_size), dim=0)
    return torch.cat(
        [chunk.narrow(0, 0, sizes[idx]).movedim(0, dim) for idx, chunk in enumerate(chunks)],
        dim=dim,
    )


def _reduce_scatter_padded_grad(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    if all(size == ctx.max_size for size in ctx.sizes):
        reduce_input = grad_output.movedim(ctx.dim, 0).contiguous()
        reduce_output = torch.empty_like(reduce_input.narrow(0, 0, ctx.max_size))
    else:
        chunks = torch.split(grad_output, ctx.sizes, dim=ctx.dim)
        padded_chunks = [
            _pad_dim(chunk, ctx.dim, ctx.max_size).movedim(ctx.dim, 0).contiguous()
            for chunk in chunks
        ]
        reduce_input = torch.cat(padded_chunks, dim=0).contiguous()
        reduce_output = torch.empty_like(padded_chunks[ctx.group_rank])
    if hasattr(dist, "reduce_scatter_tensor"):
        work = dist.reduce_scatter_tensor(
            reduce_output,
            reduce_input,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
            async_op=ctx.async_collectives,
        )
        if work is not None:
            work.wait()
    else:
        work = dist.all_reduce(
            reduce_input,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
            async_op=ctx.async_collectives,
        )
        if work is not None:
            work.wait()
        start = ctx.group_rank * ctx.max_size
        reduce_output = reduce_input.narrow(0, start, ctx.max_size).contiguous()

    grad_input = reduce_output.movedim(0, ctx.dim)
    local_size = ctx.sizes[ctx.group_rank]
    if local_size != ctx.max_size:
        grad_input = grad_input.narrow(ctx.dim, 0, local_size)
    return grad_input


class _AllReduceMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, group_size):
        ctx.group = group
        ctx.group_size = int(group_size)
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out / float(ctx.group_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input / float(ctx.group_size), None, None


class _AllGatherDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, group_size, group_rank, dim, sizes, max_size, async_collectives):
        dim = int(dim)
        if dim < 0:
            dim += x.dim()
        sizes = tuple(int(size) for size in sizes)
        max_size = int(max_size)

        ctx.group = group
        ctx.group_size = int(group_size)
        ctx.group_rank = int(group_rank)
        ctx.dim = dim
        ctx.sizes = sizes
        ctx.max_size = max_size
        ctx.async_collectives = bool(async_collectives)

        return _all_gather_padded_dim(x, group, ctx.group_size, dim, sizes, max_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _reduce_scatter_padded_grad(ctx, grad_output)
        return grad_input, None, None, None, None, None, None, None


class _AllGatherDimFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, recv, group, group_size, group_rank, dim, sizes, max_size):
        dim = int(dim)
        if dim < 0:
            dim += x.dim()
        ctx.group = group
        ctx.group_size = int(group_size)
        ctx.group_rank = int(group_rank)
        ctx.dim = dim
        ctx.sizes = tuple(int(size) for size in sizes)
        ctx.max_size = int(max_size)
        ctx.async_collectives = True
        return _assemble_gathered_buffer(recv, dim, ctx.sizes, ctx.max_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _reduce_scatter_padded_grad(ctx, grad_output)
        return grad_input, None, None, None, None, None, None, None


class _PendingAllGatherDim:
    def __init__(
        self,
        x: torch.Tensor,
        group,
        group_size: int,
        group_rank: int,
        dim: int,
        sizes: Tuple[int, ...],
        max_size: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.x = x
        self.group = group
        self.group_size = int(group_size)
        self.group_rank = int(group_rank)
        self.dim = int(dim) if dim >= 0 else int(dim) + x.dim()
        self.sizes = tuple(int(size) for size in sizes)
        self.max_size = int(max_size)
        self.output_dtype = output_dtype
        self.gathered = None
        self.send = None

        padded = _pad_dim(x, self.dim, self.max_size).contiguous()
        if TP_ALL_GATHER_INTO_TENSOR and hasattr(dist, "all_gather_into_tensor"):
            self.send = padded.movedim(self.dim, 0).contiguous()
            recv_shape = (self.group_size * self.max_size,) + tuple(self.send.shape[1:])
            self.recv = torch.empty(recv_shape, dtype=self.send.dtype, device=self.send.device)
            self.work = dist.all_gather_into_tensor(
                self.recv,
                self.send,
                group=self.group,
                async_op=True,
            )
        else:
            self.send = padded
            self.recv = None
            self.gathered = [torch.empty_like(padded) for _ in range(self.group_size)]
            self.work = dist.all_gather(
                self.gathered,
                self.send,
                group=self.group,
                async_op=True,
            )

    def wait(self) -> torch.Tensor:
        self.work.wait()
        recv = self.recv
        if recv is None:
            recv = torch.cat(
                [part.movedim(self.dim, 0) for part in self.gathered],
                dim=0,
            ).contiguous()
        if self.x.requires_grad:
            out = _AllGatherDimFromBuffer.apply(
                self.x,
                recv,
                self.group,
                self.group_size,
                self.group_rank,
                self.dim,
                self.sizes,
                self.max_size,
            )
        else:
            out = _assemble_gathered_buffer(recv, self.dim, self.sizes, self.max_size)
        if self.output_dtype is not None and out.dtype != self.output_dtype:
            out = out.to(dtype=self.output_dtype)
        return out


class TokenParallelRuntime:
    """Token Parallel runtime aligned with TokenMixer-Large data flow.

    The first RecSDK implementation keeps parameters materialized on every
    rank for checkpoint compatibility, while sharding expensive per-token
    computation across token/head dimensions and gathering activations back.
    """

    def __init__(
        self,
        enabled: bool,
        requested_size: int = 0,
        reduce_aux_loss: bool = False,
        comm_overlap: bool = False,
        comm_dtype: str = "float32",
    ) -> None:
        self.enabled = False
        self.group = None
        self.group_size = 1
        self.group_rank = 0
        self.global_rank = 0
        self.reduce_aux_loss = bool(reduce_aux_loss)
        self.comm_overlap = bool(comm_overlap)
        self.comm_dtype = str(comm_dtype).strip().lower()
        if self.comm_dtype not in ("float32", "fp32", "bfloat16", "bf16", "float16", "fp16"):
            raise ValueError("Unsupported Token Parallel communication dtype: %s" % comm_dtype)
        if not enabled or not _dist_ready():
            return

        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        tp_size = int(requested_size) if requested_size else world_size
        if tp_size <= 1:
            return
        if world_size % tp_size != 0:
            logging.warning(
                "Token Parallel disabled: world_size=%s is not divisible by token_parallel_size=%s",
                world_size,
                tp_size,
            )
            return

        selected_group = None
        for start in range(0, world_size, tp_size):
            ranks = list(range(start, start + tp_size))
            group = dist.new_group(ranks=ranks)
            if global_rank in ranks:
                selected_group = group
                self.group_rank = global_rank - start

        self.enabled = selected_group is not None
        self.group = selected_group
        self.group_size = tp_size if self.enabled else 1
        self.global_rank = global_rank
        if self.enabled and self.comm_overlap and self.global_rank == 0:
            logging.info(
                "Token Parallel full-tensor async communication overlap enabled, comm_dtype=%s",
                self.comm_dtype,
            )

    def shard_range(self, total: int) -> Tuple[int, int]:
        return _split_range(total, self.group_rank, self.group_size)

    def _prepare_comm_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.dtype]]:
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        target_dtype = dtype_map[self.comm_dtype]
        target_element_size = 4 if target_dtype == torch.float32 else 2
        if not x.is_floating_point() or x.element_size() <= target_element_size:
            return x, None
        return x.to(dtype=target_dtype), x.dtype

    def all_gather_dim(
        self,
        x: torch.Tensor,
        dim: int,
        total_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int], int]:
        if not self.enabled:
            return x, [x.size(dim)], 0

        if total_size is None:
            local_size = torch.tensor([x.size(dim)], device=x.device, dtype=torch.int64)
            gathered_sizes = [torch.empty_like(local_size) for _ in range(self.group_size)]
            dist.all_gather(gathered_sizes, local_size, group=self.group)
            sizes = [int(size.item()) for size in gathered_sizes]
        else:
            sizes = [
                _split_range(int(total_size), rank, self.group_size)[1]
                - _split_range(int(total_size), rank, self.group_size)[0]
                for rank in range(self.group_size)
            ]
        offsets = [0]
        for size in sizes[:-1]:
            offsets.append(offsets[-1] + size)

        max_size = max(sizes)
        comm_x, output_dtype = self._prepare_comm_input(x)
        if comm_x.requires_grad:
            gathered = _AllGatherDim.apply(
                comm_x,
                self.group,
                self.group_size,
                self.group_rank,
                dim,
                tuple(sizes),
                max_size,
                self.comm_overlap,
            )
        else:
            gathered = _all_gather_padded_dim(
                comm_x,
                self.group,
                self.group_size,
                dim,
                tuple(sizes),
                max_size,
            )
        if output_dtype is not None and gathered.dtype != output_dtype:
            gathered = gathered.to(dtype=output_dtype)
        return gathered, sizes, offsets[self.group_rank]

    def begin_all_gather_dim(
        self,
        x: torch.Tensor,
        dim: int,
        total_size: int,
    ) -> _PendingAllGatherDim:
        if not self.enabled:
            raise RuntimeError("begin_all_gather_dim requires Token Parallel to be enabled")
        sizes = tuple(
            _split_range(int(total_size), rank, self.group_size)[1]
            - _split_range(int(total_size), rank, self.group_size)[0]
            for rank in range(self.group_size)
        )
        comm_x, output_dtype = self._prepare_comm_input(x)
        return _PendingAllGatherDim(
            comm_x,
            self.group,
            self.group_size,
            self.group_rank,
            dim,
            sizes,
            max(sizes),
            output_dtype=output_dtype,
        )

    def should_overlap(self, x: torch.Tensor) -> bool:
        if not self.enabled or not self.comm_overlap:
            return False
        if x.device.type not in ("npu", "cuda"):
            return False
        return True

    def all_reduce_mean(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return _AllReduceMean.apply(x, self.group, self.group_size)


def _make_norm(dim: int, eps: float, use_rmsnorm: bool) -> nn.Module:
    if use_rmsnorm:
        return RMSNormNPU(dim, eps)
    return nn.LayerNorm(dim, eps=eps)


class PerTokenSwiGLU(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        npu_bmm: bool = True,
        fused_gate_up: bool = True,
        use_npu_swiglu: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(dim * hidden_multiplier)))
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.npu_bmm = npu_bmm
        self.fused_gate_up = fused_gate_up
        self.use_npu_swiglu = use_npu_swiglu
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.gate_up_weight = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim * 2))
        self.down_weight = nn.Parameter(torch.empty(num_tokens, hidden_dim, dim))
        if bias:
            self.gate_up_bias = nn.Parameter(torch.empty(num_tokens, hidden_dim * 2))
            self.down_bias = nn.Parameter(torch.empty(num_tokens, dim))
        else:
            self.register_parameter("gate_up_bias", None)
            self.register_parameter("down_bias", None)
        self.down_init_scale = down_init_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
        nn.init.xavier_normal_(gate_weight)
        nn.init.xavier_normal_(up_weight)
        nn.init.xavier_normal_(self.down_weight)
        if self.down_init_scale != 1.0:
            with torch.no_grad():
                self.down_weight.mul_(self.down_init_scale)
        if self.gate_up_bias is not None:
            nn.init.zeros_(self.gate_up_bias)
            nn.init.zeros_(self.down_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self._gate_up(x)
        if self.gate_up_bias is not None:
            gate_up = gate_up + self.gate_up_bias
        y = self._swiglu(gate_up)
        y = self.dropout(y)
        y = self._per_token_linear(y, self.down_weight)
        if self.down_bias is not None:
            y = y + self.down_bias
        return self.dropout(y)

    def _forward_with_params(
        self,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_up_bias: Optional[torch.Tensor],
        down_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.npu_bmm:
            gate_up = torch.bmm(x.transpose(0, 1), gate_up_weight).transpose(0, 1)
        else:
            gate_up = torch.einsum("btd,tdh->bth", x, gate_up_weight)
        if gate_up_bias is not None:
            gate_up = gate_up + gate_up_bias
        y = self._swiglu(gate_up)
        y = self.dropout(y)
        if self.npu_bmm:
            y = torch.bmm(y.transpose(0, 1), down_weight).transpose(0, 1)
        else:
            y = torch.einsum("bth,thd->btd", y, down_weight)
        if down_bias is not None:
            y = y + down_bias
        return self.dropout(y)

    def forward_shard(self, x: torch.Tensor, token_start: int) -> torch.Tensor:
        local_tokens = x.size(1)
        if local_tokens == 0:
            return x.new_zeros(x.shape)
        token_end = token_start + local_tokens
        gate_up_bias = self.gate_up_bias[token_start:token_end] if self.gate_up_bias is not None else None
        down_bias = self.down_bias[token_start:token_end] if self.down_bias is not None else None
        return self._forward_with_params(
            x,
            self.gate_up_weight[token_start:token_end],
            self.down_weight[token_start:token_end],
            gate_up_bias,
            down_bias,
        )

    def _gate_up(self, x: torch.Tensor) -> torch.Tensor:
        if self.npu_bmm:
            xt = x.transpose(0, 1)
            if self.fused_gate_up:
                return torch.bmm(xt, self.gate_up_weight).transpose(0, 1)
            gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
            gate = torch.bmm(xt, gate_weight).transpose(0, 1)
            up = torch.bmm(xt, up_weight).transpose(0, 1)
            return torch.cat([gate, up], dim=-1)
        gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
        gate = torch.einsum("btd,tdh->bth", x, gate_weight)
        up = torch.einsum("btd,tdh->bth", x, up_weight)
        return torch.cat([gate, up], dim=-1)

    def _swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        npu_swiglu = None
        if self.use_npu_swiglu and gate_up.device.type == "npu":
            npu_swiglu = _get_npu_op("npu_swiglu")
        if npu_swiglu is not None:
            return npu_swiglu(gate_up, dim=-1)
        gate, up = gate_up.split(self.hidden_dim, dim=-1)
        return F.silu(gate) * up

    def _per_token_linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.npu_bmm:
            return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
        return torch.einsum("bth,thd->btd", x, weight)

    def forward_selected(
        self,
        x: torch.Tensor,
        flat_indices: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_indices = torch.arange(self.num_tokens, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        flat_token_indices = token_indices.reshape(-1).index_select(0, flat_indices)
        selected = x.reshape(-1, self.dim).index_select(0, flat_indices).unsqueeze(1)
        gate_up_weight = self.gate_up_weight.index_select(0, flat_token_indices)
        gate_up = torch.bmm(selected, gate_up_weight).squeeze(1)
        if self.gate_up_bias is not None:
            gate_up = gate_up + self.gate_up_bias.index_select(0, flat_token_indices)
        y = self._swiglu(gate_up)
        y = self.dropout(y)
        down_weight = self.down_weight.index_select(0, flat_token_indices)
        y = torch.bmm(y.unsqueeze(1), down_weight).squeeze(1)
        if self.down_bias is not None:
            y = y + self.down_bias.index_select(0, flat_token_indices)
        if weights is not None:
            y = y * weights.unsqueeze(-1)
        return self.dropout(y)


class LegacyPerTokenFFN(nn.Module):
    """TokenMixer-Large fallback for ablation only: per-token FFN without SwiGLU."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        npu_bmm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(dim * hidden_multiplier)))
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.npu_bmm = npu_bmm
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.up_weight = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim))
        self.down_weight = nn.Parameter(torch.empty(num_tokens, hidden_dim, dim))
        if bias:
            self.up_bias = nn.Parameter(torch.empty(num_tokens, hidden_dim))
            self.down_bias = nn.Parameter(torch.empty(num_tokens, dim))
        else:
            self.register_parameter("up_bias", None)
            self.register_parameter("down_bias", None)
        self.down_init_scale = down_init_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.up_weight)
        nn.init.xavier_normal_(self.down_weight)
        if self.down_init_scale != 1.0:
            with torch.no_grad():
                self.down_weight.mul_(self.down_init_scale)
        if self.up_bias is not None:
            nn.init.zeros_(self.up_bias)
            nn.init.zeros_(self.down_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.npu_bmm:
            y = torch.bmm(x.transpose(0, 1), self.up_weight).transpose(0, 1)
        else:
            y = torch.einsum("btd,tdh->bth", x, self.up_weight)
        if self.up_bias is not None:
            y = y + self.up_bias
        y = F.gelu(y)
        y = self.dropout(y)
        if self.npu_bmm:
            y = torch.bmm(y.transpose(0, 1), self.down_weight).transpose(0, 1)
        else:
            y = torch.einsum("bth,thd->btd", y, self.down_weight)
        if self.down_bias is not None:
            y = y + self.down_bias
        return self.dropout(y)

    def _forward_with_params(
        self,
        x: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        up_bias: Optional[torch.Tensor],
        down_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.npu_bmm:
            y = torch.bmm(x.transpose(0, 1), up_weight).transpose(0, 1)
        else:
            y = torch.einsum("btd,tdh->bth", x, up_weight)
        if up_bias is not None:
            y = y + up_bias
        y = F.gelu(y)
        y = self.dropout(y)
        if self.npu_bmm:
            y = torch.bmm(y.transpose(0, 1), down_weight).transpose(0, 1)
        else:
            y = torch.einsum("bth,thd->btd", y, down_weight)
        if down_bias is not None:
            y = y + down_bias
        return self.dropout(y)

    def forward_shard(self, x: torch.Tensor, token_start: int) -> torch.Tensor:
        local_tokens = x.size(1)
        if local_tokens == 0:
            return x.new_zeros(x.shape)
        token_end = token_start + local_tokens
        up_bias = self.up_bias[token_start:token_end] if self.up_bias is not None else None
        down_bias = self.down_bias[token_start:token_end] if self.down_bias is not None else None
        return self._forward_with_params(
            x,
            self.up_weight[token_start:token_end],
            self.down_weight[token_start:token_end],
            up_bias,
            down_bias,
        )
