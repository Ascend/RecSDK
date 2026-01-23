# ExpandIntoJaggedPermute

本算子仅支持 NPU 调用。

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A5 训练系列产品|√|

# 目录结构
```shell
-- expand_into_jagged_permute
   |-- c310
      |-- op_host                                   # 算子 host 侧实现
      |-- op_kernel                                 # 算子 kernel 侧实现
      |-- expand_into_jagged_permute.json
      |-- README.md
      |-- run.sh
```

## 功能说明

- 算子功能：将稀疏数据置换索引从表维度扩展到批次维度，适用于稀疏特征在不同rank中具有不同批次大小的情况。
- 计算公式：

对于每个位置 i (0 ≤ i < permute.numel())：

$$
len = outputOffset[i+1] - outputOffset[i]
$$

$$
outputPermuteOut[outputOffset[i]:outputOffset[i+1]] = arange(inputOffset[permute[i]], inputOffset[permute[i]+1])
$$

其中 `len = inputOffset[permute[i]+1] - inputOffset[permute[i]]`，即第 `permute[i]` 个表的长度。

- Python 伪代码实现：

```python
def expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size):
    """
    将表级别的置换索引扩展到批次维度。
    
    Args:
        permute: 表级别的置换索引，形状为 [num_tables]
        input_offsets: 输入表的累积偏移量，形状为 [num_tables + 1]
        output_offsets: 输出表的累积偏移量，形状为 [num_tables + 1]
        output_size: 输出结果的长度
    
    Returns:
        output_permute: 扩展后的置换索引，形状为 [output_size]
    """
    output_permute = []
    for i in range(len(permute)):
        # 获取第 permute[i] 个表的起始和结束位置
        start_idx = input_offsets[permute[i]]
        end_idx = input_offsets[permute[i] + 1]
        # 生成该表的索引序列并追加到输出
        output_permute.extend(range(start_idx, end_idx))
    return output_permute
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>permute</td>
      <td>输入</td>
      <td>表示表级别的置换索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputOffset</td>
      <td>输入</td>
      <td>表示表级别长度的互斥偏移量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputOffset</td>
      <td>输入</td>
      <td>表示表级别置换长度的互斥偏移量。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputPermute</td>
      <td>输出</td>
      <td>表示公式中的输出。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>outputSize</td>
      <td>属性</td>
      <td>输出结果的长度。</td>
      <td>INT64</td>
      <td>NA</td>
    </tr>
  </tbody></table>

## 约束说明
- inputOffset、outputOffset的长度比permute多1。
- 用户需用户自行保证输入、输出长度与大小不超过对应数据类型的数值上限。


## 编译与部署
参考 RecSDK/cust_op/README.md “单算子使用说明”章节的编译、适配层部署流程。

更多 PyTorch 调用示例见 framework/torch_plugin/torch_library/expand_into_jagged_permute/README.md。
