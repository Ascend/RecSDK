# 使用 pytorch 框架调用方式调用 gen_position_ids_reverse_v2 算子

该样例基于 Pytorch2.6.0、python3.11.0 运行

## Pytorch 框架对外接口原型

```python
torch.ops.fbgemm.gen_position_ids_reverse_v2(
    Tensor seqlen,
    Tensor seqlen_offsets,
    Tensor rspos,
    int batch_size,
    bool interleaved_action=False,
    bool with_ctx=False
) -> Tensor
或
torch.ops.mxrec.gen_position_ids_reverse_v2(
    Tensor seqlen,
    Tensor seqlen_offsets,
    Tensor rspos,
    int batch_size,
    bool interleaved_action=False,
    bool with_ctx=False
) -> Tensor
```

## 参数说明

| 名称               | 输入/输出 | 参数类型 | 数据类型 | 数据格式                                      | 说明 |
|--------------------|-----------|----------|----------|-----------------------------------------------|------|
| seqlen             | 输入      | Tensor   | int32    | 一维，长度 = batch                            | 每个样本的序列长度 |
| seqlen_offsets     | 输入      | Tensor   | int32    | 一维，长度 = batch + 1，前缀/累积偏移         | 第 i 条在扁平 position_ids 中的起始下标为 offsets[i] |
| rspos              | 输入      | Tensor   | int32    | 一维，长度 = batch                            | 每条序列中「前半段」长度（历史/前置段），后半填 0 |
| batch_size         | 输入      | int      | int64    | ≥ 0，且须等于 `seqlen.size(0)`               | 批量大小，与 aclnn 侧 `batchSize` 属性一致 |
| interleaved_action | 输入      | bool     | bool     | NPU 实现不支持 `True`                         | 与 CUDA/Python 接口对齐；为 `True` 时将 `TORCH_CHECK` 失败 |
| with_ctx           | 输入      | bool     | bool     | NPU 实现不支持 `True`                         | 与 CUDA/Python 接口对齐；为 `True` 时将 `TORCH_CHECK` 失败 |
| position_ids       | 输出      | Tensor   | int32    | 一维，长度 = `seqlen_offsets[batch_size]`     | 按 jagged 展平后的位置 id |

**算子分析**

该算子在变长（jagged）序列上生成 reverse v2 风格的一维 `position_ids`：对第 `i` 条样本，设 `seq_len = seqlen[i]`，`rs = rspos[i]`，在 `seqlen_offsets[i]` 起的连续 `seq_len` 个位置上，前 `rs` 个为 `rs, rs-1, …, 1`，其余为 `0`。与将「整段简单倒序」的 v1 不同，v2 体现同一行为段共享递降位置、尾部填零的语义。NPU 路径通过 `aclnnGenPositionIdsReverseV2` 调用自定义算子，参见 Ascend 侧 [README.md](../../../../ascendc_op/ai_core_op/gen_position_ids_reverse_v2/c310/README.md)。

## 运行算子样例

### 算子调用示例（pytest）

```python
import torch
import torch_npu


def custom_op_exec(seqlen, seqlen_offsets, rspos, batch_size, interleaved_action=False, with_ctx=False):
    seqlen_npu = seqlen.to("npu")
    seqlen_offsets_npu = seqlen_offsets.to("npu")
    rspos_npu = rspos.to("npu")
    position_ids = torch.ops.fbgemm.gen_position_ids_reverse_v2(
        seqlen_npu,
        seqlen_offsets_npu,
        rspos_npu,
        batch_size,
        interleaved_action,
        with_ctx,
    )
    torch.npu.synchronize()
    return position_ids.cpu()


# 使用前需按工程说明 load_library 加载 fbgemm_npu_api.so；batch_size 与 seqlen 行数一致。
```
