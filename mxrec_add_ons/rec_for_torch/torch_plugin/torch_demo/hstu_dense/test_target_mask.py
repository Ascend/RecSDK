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


def _is_this_point_in_context(row_on_score, num_context, col_on_score, num_history):
    return row_on_score < num_context and col_on_score < num_history


def _is_this_point_out_border(row_on_score, col_on_score, seq_len):
    return row_on_score >= seq_len or col_on_score >= seq_len


def _is_this_point_in_casual_mask(row_on_score, col_on_score):
    return col_on_score > row_on_score


def _is_this_point_in_target_mask(
    row_on_score, col_on_score, num_history, target_group_size
):
    if row_on_score >= num_history and col_on_score > num_history:
        target_index = (row_on_score - num_history) // target_group_size
        target_col_limit_left = num_history + target_index * target_group_size
        if col_on_score < target_col_limit_left:
            return True
    return False


def _compute_target_mask_one_block(
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
            if _is_this_point_out_border(row_on_score, col_on_score, param.seq_len):
                continue
            if _is_this_point_in_context(
                row_on_score, param.num_context, col_on_score, param.num_history
            ):
                continue
            if _is_this_point_in_casual_mask(row_on_score, col_on_score):
                block_mask[row_id_on_block][col_id_on_block] = 0
            if _is_this_point_in_target_mask(
                row_on_score, col_on_score, param.num_history, param.target_group_size
            ):
                block_mask[row_id_on_block][col_id_on_block] = 0
    return block_mask


def compute_target_mask_each_block(
    score_shape_param: ScoreShapeParam,
) -> list[torch.Tensor]:
    block_num_on_q = (
        score_shape_param.seq_len + score_shape_param.block_h - 1
    ) // score_shape_param.block_h
    block_num_on_k = (
        score_shape_param.seq_len + score_shape_param.block_w - 1
    ) // score_shape_param.block_w
    score_mask_blocks = []

    for block_id_q in range(block_num_on_q):
        blocks_on_one_k_line = []
        for block_id_k in range(block_num_on_k):
            block_param = HstuBlockParam(
                block_id_q,
                block_id_k,
                score_shape_param.block_h,
                score_shape_param.block_w,
            )
            block_mask = _compute_target_mask_one_block(block_param, score_shape_param)
            blocks_on_one_k_line.append(block_mask)
        blocks_on_one_k_line_concat = torch.concat(blocks_on_one_k_line, dim=1)
        score_mask_blocks.append(blocks_on_one_k_line_concat)
    score_mask = torch.concat(score_mask_blocks, dim=0)
    score_mask_without_padding = score_mask[
        : score_shape_param.seq_len, : score_shape_param.seq_len
    ]
    return score_mask_without_padding


def write_tensor2file(tensor: torch.Tensor):
    tensor_list = tensor.long().tolist()
    with open("target_mask_tesnor_example.txt", "w") as f:
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
    result = compute_target_mask_each_block(score_shape_param)
    write_tensor2file(result)


if __name__ == "__main__":
    reuslt = test_hstu_target_mask(TestParam(64, 16, 16, 4, 8, 8))
