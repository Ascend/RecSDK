# 使用pytorch框架调用方式调用gen_position_ids_with_timestamp算子

该样例基于Pytorch2.6.0、python3.11.0运行

## Pytorch框架对外接口原型

```python
torch.ops.mxrec.gen_position_ids_with_timestamp(
    Tensor seqlen,
    Tensor seqlen_offsets,
    Tensor timestamps,
    int batch_size,
    int total_seq_len,
    float? time_scale=300.0
) -> Tensor
```

## 参数说明

| 名称            | 输入/输出 | 参数类型 |  数据类型  | 数据格式                                       | 范围             | 说明                          |
|---------------|-------|  ----  |  ----  |--------------------------------------------|----------------|-----------------------------|
| seqlen        | 输入    | Tensor | int32  | torch.tensor([value1, value2, ...])       | 长度:[1, 65536] | 每个样本的序列长度                |
| seqlen_offsets| 输入    | Tensor | int32  | torch.tensor([0, value1, value2, ...])    | 长度:[1, 65537] | 每个样本在 timestamps 中的起始位置   |
| timestamps    | 输入    | Tensor | int32  | torch.tensor([t1, t2, t3 ...])            |                | 时间戳数组                      |
| batch_size    | 输入    | int    | int64  |                                           |                | 批量大小                       |
| total_seq_len | 输入    | int    | int64  |                                           |                | 总序列长度                     |
| time_scale    | 输入    | float  | float  |                                           | > 0            | 时间缩放因子（可选，默认 300.0）     |
| position_ids  | 输出    | Tensor | int32  | torch.tensor([pos1, pos2, ...])           | [0, 1024]      | 位置编码                       |

**算子分析**

该算子根据时间戳生成位置编码，核心公式如下：

```python
pos = log1p((t_end - tm) / time_scale) / log(log_base)
```

其中 log_base = 1.1，position_id 范围限制为 [0, 1024]。

## 运行算子样例

### 算子调用示例,以下以pytest方式调用为例

```python
import pytest
import torch
import torch_npu
import math

# 默认 time_scale
DEFAULT_TIME_SCALE = 300.0


def compute_position_ids_golden(seqlen, seqlen_offsets, timestamps, time_scale):
    """Golden 参考实现"""
    inv_log_base = 10.4920586873  # 1 / ln(1.1)
    max_position_id = 1024

    batch_size = seqlen.shape[0]
    total_len = timestamps.shape[0]
    position_ids = torch.zeros(total_len, dtype=torch.int32)

    for batch_idx in range(batch_size):
        seq_len = seqlen[batch_idx].item()
        start_pos = seqlen_offsets[batch_idx].item()
        end_pos = seqlen_offsets[batch_idx + 1].item()

        t_end = timestamps[end_pos - 1].item()

        for offset in range(seq_len):
            timestamp_idx = start_pos + offset
            timestamp = timestamps[timestamp_idx].item()

            time_diff = (t_end - timestamp) / time_scale
            log_pos = math.log(1.0 + time_diff) * inv_log_base

            position_id = max(0, min(int(math.floor(log_pos)), max_position_id))
            position_ids[timestamp_idx] = position_id

    return position_ids


def jagged_data_gen(batch_size, max_seq_len):
    """生成测试数据"""
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_offset = torch.zeros(batch_size + 1, dtype=torch.int32)
    seq_offset[1:] = torch.cumsum(seq_lens, dim=0)

    total_len = seq_offset[-1].item()

    # 时间戳生成：每个 batch 内的 timestamps 需要递增
    timestamps_list = []
    for batch_idx in range(batch_size):
        seq_len = seq_lens[batch_idx].item()
        base = torch.randint(0, 5000, (1,)).item()
        timestamps_batch = torch.arange(base, base + seq_len, dtype=torch.int32)
        timestamps_list.append(timestamps_batch)

    timestamps = torch.cat(timestamps_list)

    return seq_lens, seq_offset, timestamps


def test_gen_position_ids_with_timestamp():
    batch_size = 4
    max_seq_len = 128
    time_scale = 300.0

    seqlen, seqlen_offsets, timestamps = jagged_data_gen(batch_size, max_seq_len)
    total_seq_len = timestamps.shape[0]

    # 调用算子
    position_ids = torch.ops.mxrec.gen_position_ids_with_timestamp(
        seqlen.npu(),
        seqlen_offsets.npu(),
        timestamps.npu(),
        batch_size,
        total_seq_len,
        time_scale
    ).cpu()

    # Golden 计算
    golden = compute_position_ids_golden(seqlen, seqlen_offsets, timestamps, time_scale)

    # 验证
    assert torch.allclose(position_ids, golden, atol=1)


if __name__ == "__main__":
    test_gen_position_ids_with_timestamp()
    print("Test passed!")
```
