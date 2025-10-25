# relative_attn_bias_pos优化器融合算子及样例说明
本算子仅支持NPU调用

## relative_attn_bias_pos融合算子文件结构

```shell
├── relative_attn_bias_pos.json    # 算子原型配置
├── op_host    # relative_attn_bias_pos融合算子Host侧实现
├── op_kernel  # relative_attn_bias_pos融合算子Kernel侧实现
├── README.md  # relative_attn_bias_pos融合算子说明文档
└── run.sh     # relative_attn_bias_pos融合算子安装脚本
```

## relative_attn_bias_pos融合算子介绍

1. 算子分析

a) 算子参数说明：

| 算子参数              | 输入/输出 | dtype     | shape       |
|-------------------|-------|-----------|-------------|
| rel_pos_bias      | 输入    | FP16,FP32 | (2s, 2s)    |
| identity          | 输入    | FP16,FP32 | (2s, 2s)    |
| past_valid_lens   | 输入    | List[int] | (b,)        |
| rab_pos           | 输出    | FP16,FP32 | (b, 2s, 2s) |


b) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；

## 算子逻辑

```python
import torch


def rab_pos_golden(rel_pos_bias: torch.Tensor, identity: torch.Tensor, past_valid_lens: torch.Tensor) -> torch.Tensor:
    """
    past_len = 1 ~ 4000
    candidate_len = 256 ~ 600
    bs = 1 ~ 10

    :param rel_pos_bias: [past_len * 2 + candidate_len][past_len * 2 + candidate_len]
    :param identity: [past_len * 2 + candidate_len][past_len * 2 + candidate_len]
    :param past_valid_lens: [bs]
    :return: [bs][1][past_len * 2 + candidate_len + 2][past_len * 2 + candidate_len + 2]
    """
    bs = past_valid_lens.shape[0]
    rel_pos_bias_list = rel_pos_bias[:].unsqueeze(0).repeat(bs, 1, 1)
    for i, valid_len in enumerate(past_valid_lens):
        rel_pos_bias_list[i, valid_len:, :] = rel_pos_bias[valid_len, :]

    rel_pos_bias_list = rel_pos_bias_list * (1 - identity) + identity * rel_pos_bias_list[0, 0, 0]
    rel_pos_bias_list = rel_pos_bias_list[:, :identity.shape[0], :identity.shape[0]]
    return rel_pos_bias_list
```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)
