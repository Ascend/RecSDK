from dataclasses import dataclass
import itertools

import pytest
import torch


@dataclass
class HstuBlockParam:
    block_id_q: int = 0
    block_id_k: int = 0
    block_h: int = 0
    block_w: int = 0


@dataclass
class ScoreShapeParam:
    seq_len: int = 0
    num_target: int = 0
    num_context: int = 0
    num_history: int = 0
    target_group_size: int = 0
    block_h: int = 0
    block_w: int = 0


def _check_param_valid(seq_len, num_target, num_context, target_group_size) -> bool:
    if seq_len < num_target + num_context:
        return False
    if target_group_size > num_target:
        return False
    return True


class GPUSmit:
    @staticmethod
    def is_this_point_in_context(row_on_score, num_context, col_on_score, num_history):
        return row_on_score < num_context and col_on_score < num_history

    @staticmethod
    def is_this_point_out_border(row_on_score, col_on_score, seq_len):
        return row_on_score >= seq_len or col_on_score >= seq_len

    @staticmethod
    def is_this_point_in_casual_mask(row_on_score, col_on_score):
        return col_on_score > row_on_score

    @staticmethod
    def is_this_point_in_target_mask(
        row_on_score, col_on_score, num_history, target_group_size
    ):
        if row_on_score >= num_history and col_on_score >= num_history:
            target_index = (row_on_score - num_history) // target_group_size
            target_col_limit_left = num_history + target_index * target_group_size
            if col_on_score < target_col_limit_left:
                return True
        return False


def _duplicate(ub: torch.Tensor, value, mask_len):
    ub[:mask_len] = value


def _compute_col_start_and_end_on_score(block_id_k, block_w):
    col_on_score_range = [block_id_k * block_w, (block_id_k + 1) * block_w] 
    return col_on_score_range


def _is_this_line_on_context(row_on_score, param: ScoreShapeParam):
    return row_on_score < param.num_context


def _process_line_on_context(
    block_mask_this_line, col_on_score_range: list[int], param: ScoreShapeParam
):
    """
    从后向前看需要有多少个mask
    """
    col_end = col_on_score_range[1]
    mask_len = col_end - param.num_history
    if mask_len > 0:
        _duplicate(block_mask_this_line[-mask_len:], 0, mask_len)


def _process_line_with_causal(
    block_mask_this_line, row_on_score, col_on_score_range, param: ScoreShapeParam
):
    """
    从后向前看需要有多少个mask
    """
    col_end_on_score = col_on_score_range[1]
    mask_len = col_end_on_score - row_on_score - 1
    if mask_len > 0:
        _duplicate(block_mask_this_line[-mask_len:], 0, mask_len)


def process_one_block_of_target_mask(
    block_mask, block_param: HstuBlockParam, param: ScoreShapeParam
):
    col_on_score_range = _compute_col_start_and_end_on_score(
        block_param.block_id_k, block_param.block_w
    )
    if col_on_score_range[1] <= param.num_history:
        return
    mask_start_in_block = (
        0
        if col_on_score_range[0] > param.num_history
        else param.num_history - col_on_score_range[0]
    )
    mask_col_start_in_score = (
        col_on_score_range[0]
        if col_on_score_range[0] > param.num_history
        else param.num_history
    )

    for row_id_on_block in range(block_param.block_h):
        row_on_score = row_id_on_block + block_param.block_id_q * block_param.block_h
        target_mask_end_on_score = (
            row_on_score - param.num_history
        ) // param.target_group_size * param.target_group_size + param.num_history
        # 这一行还没有遇到target mask
        # 这一行target mask的末尾比起始位置长
        if (
            row_on_score < param.num_history + param.target_group_size
            or target_mask_end_on_score <= col_on_score_range[0]
        ):
            continue
        mask_len = (
            min(col_on_score_range[1], target_mask_end_on_score) - mask_col_start_in_score
        )
        block_mask_this_line = block_mask[row_id_on_block, :]
        _duplicate(block_mask_this_line[mask_start_in_block:], 0, mask_len)


def _compute_target_mask_one_block_npu(
    block_param: HstuBlockParam, param: ScoreShapeParam
) -> torch.Tensor:
    """
    初始化mask为全1,遍历每个point, 经过context mask, casual mask, target mask. 扣去其中为0的部分
    """
    block_mask = torch.ones((block_param.block_h, block_param.block_w))
    col_on_score_range = _compute_col_start_and_end_on_score(
        block_param.block_id_k, block_param.block_w
    )
    for row_id_on_block in range(block_param.block_h):
        row_on_score = row_id_on_block + block_param.block_id_q * block_param.block_h
        block_mask_this_line = block_mask[row_id_on_block, :]
        if _is_this_line_on_context(row_on_score, param):
            _process_line_on_context(block_mask_this_line, col_on_score_range, param)
        else:
            # 滿足causul的条件一定不满足在context
            _process_line_with_causal(
                block_mask_this_line, row_on_score, col_on_score_range, param
            )

    process_one_block_of_target_mask(block_mask, block_param, param)
    return block_mask


def _compute_target_mask_one_block_gpu(
    block_param: HstuBlockParam, param: ScoreShapeParam
) -> torch.Tensor:
    """
    初始化mask为全1,遍历每个point, 经过context mask, casual mask, target mask. 扣去其中为0的部分
    """
    block_mask = torch.ones((block_param.block_h, block_param.block_w))

    for row_id_on_block in range(block_param.block_h):
        for col_id_on_block in range(block_param.block_w):
            row_on_score = (
                row_id_on_block + block_param.block_id_q * block_param.block_h
            )
            col_on_score = (
                col_id_on_block + block_param.block_id_k * block_param.block_w
            )
            if GPUSmit.is_this_point_in_context(
                row_on_score, param.num_context, col_on_score, param.num_history
            ):
                continue
            if GPUSmit.is_this_point_in_casual_mask(row_on_score, col_on_score):
                block_mask[row_id_on_block][col_id_on_block] = 0
            if GPUSmit.is_this_point_in_target_mask(
                row_on_score, col_on_score, param.num_history, param.target_group_size
            ):
                block_mask[row_id_on_block][col_id_on_block] = 0
    return block_mask


def compute_target_mask_each_block_concat(
    score_shape_param: ScoreShapeParam, use_npu
) -> list[torch.Tensor]:
    block_num_on_q = (
        score_shape_param.seq_len + score_shape_param.block_h - 1
    ) // score_shape_param.block_h
    block_num_on_k = (
        score_shape_param.seq_len + score_shape_param.block_w - 1
    ) // score_shape_param.block_w
    score_mask_blocks = [
        [None for _ in range(block_num_on_k)] for _ in range(block_num_on_q)
    ]
    for block_id_q in range(block_num_on_q):
        for block_id_k in range(block_num_on_k):
            block_param = HstuBlockParam(
                block_id_q,
                block_id_k,
                score_shape_param.block_h,
                score_shape_param.block_w,
            )
            if not use_npu:
                block_mask = _compute_target_mask_one_block_gpu(
                    block_param, score_shape_param
                )
            else:
                block_mask = _compute_target_mask_one_block_npu(
                    block_param, score_shape_param
                )
            score_mask_blocks[block_id_q][block_id_k] = block_mask

        score_mask_blocks[block_id_q] = torch.concat(
            score_mask_blocks[block_id_q], dim=1
        )
    score_mask = torch.concat(score_mask_blocks, dim=0)
    score_mask_without_padding = score_mask[
        : score_shape_param.seq_len, : score_shape_param.seq_len
    ]
    return score_mask_without_padding


def write_tensor2file(tensor: torch.Tensor):
    tensor_list = tensor.long().tolist()
    with open("target_mask_tesnor_example_gen.txt", "w") as f:
        for i in range(tensor.shape[0]):
            one_line_str = ",".join([str(mask) for mask in tensor_list[i]])
            f.write(one_line_str + "\n")


test_param_all = {
    "seq_len": [64],
    "num_target": [16],
    "num_context": [16],
    "target_group_size": [4],
    "block_height": [8],
    "block_weight": [8],
}


@dataclass
class TestParam:
    seq_len: int
    num_target: int
    num_context: int
    target_group_size: int
    block_height: int
    block_weight: int


@pytest.mark.parametrize(
    "test_param", [TestParam(*v) for v in itertools.product(*test_param_all.values())]
)
def test_hstu_target_mask(test_param: TestParam):
    seq_len = test_param.seq_len
    num_target = test_param.num_target
    num_context = test_param.num_context
    target_group_size = test_param.target_group_size
    block_height = test_param.block_height
    block_weight = test_param.block_weight

    is_valid = _check_param_valid(seq_len, num_target, num_context, target_group_size)
    if not is_valid:
        raise RuntimeError("param is not valid")
    num_history = seq_len - num_target
    score_shape_param = ScoreShapeParam(
        seq_len,
        num_target,
        num_context,
        num_history,
        target_group_size,
        block_height,
        block_weight,
    )
    result_gpu = compute_target_mask_each_block_concat(score_shape_param, use_npu=False)
    result_npu = compute_target_mask_each_block_concat(score_shape_param, use_npu=True)
    write_tensor2file(result_npu)
    assert torch.allclose(
        result_gpu, result_npu, 1e-4, 1e-4
    ), f"gloden {result_gpu} result {result_npu} not close"


if __name__ == "__main__":
    reuslt = test_hstu_target_mask(TestParam(65, 31, 17, 5, 8, 8))
