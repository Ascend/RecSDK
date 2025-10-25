# relative_attn_bias_time优化器融合算子及样例说明
本算子仅支持NPU调用

## relative_attn_bias_time融合算子文件结构

```shell
├── relative_attn_bias_time.json    # 算子原型配置
├── op_host    # relative_attn_bias_time融合算子Host侧实现
├── op_kernel  # relative_attn_bias_time融合算子Kernel侧实现
├── README.md  # relative_attn_bias_time融合算子说明文档
└── run.sh     # relative_attn_bias_time融合算子安装脚本
```


## relative_attn_bias_time融合算子介绍

1. 算子分析

a) 算子参数说明：

| 算子参数               | 输入/输出 | dtype     | shape                       |
|--------------------|-------|-----------|-----------------------------|
| timestamps         | 输入    | int32     | (b, s)                      |
| timestamps_weights | 输入    | FP16,FP32 | (num_layers, num_buckets+1) |
| bucket_divisor     | 输入    | float     |                             |
| rab_time           | 输出    | FP16,FP32 | (num_layers, b, s, 1, s, 1) |
| bucket_timestamps  | 输出    | int32     | (b, s, s)                   |
> 注：rab_time期望返回为(num_layers, b, 2s, 2s)  
> ONNX调用时需进行rab_time.repeat(1, 1, 1, 2, 1, 2).reshape(num_layers, b, 2s, 2s)操作。  
> pytorch调用无需额外操作，已经在pytorch适配层完成该操作。

b) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；

## 算子逻辑

```python
import torch
NUM_BUCKETS = 128 + 1
BUCKET_DIVISOR = 0.301

def rab_time_golden(timestamps_weights: torch.Tensor,
                    timestamps: torch.Tensor,
                    bucket_divisor: float) -> (torch.Tensor, torch.Tensor):
    """
    rab time 正向仿真
    num_buckets = 128
    num_layers = 1 - 20
    past_len = 1 - 4000
    candidate_len = 256 - 600

    :param timestamps_weights: [num_buckets + 1][num_layers]
    :param timestamps: [bs][past_len + candidate_len // 2]
    :param bucket_divisor: float
    :return: [num_layers][bs][1][2 * past_len + candidate_len + 1][2 * past_len + candidate_len + 2]
    """

    infer_len = timestamps.shape[1] * 2
    bs = timestamps.shape[0]
    num_layers = timestamps_weights.shape[1]

    timestamps = timestamps.unsqueeze(-1).repeat(1, 1, 2)
    diff_timestamps = timestamps.reshape(bs, infer_len, 1) - timestamps.reshape(bs, 1, infer_len)

    clamp_max = torch.exp(torch.tensor(NUM_BUCKETS * BUCKET_DIVISOR))
    diff_timestamps = torch.log(torch.abs(diff_timestamps).clamp(1, clamp_max)) / bucket_divisor

    bucket_timestamps = diff_timestamps.long().view(-1)
    rab_time_out = torch.index_select(timestamps_weights, dim=0, index=bucket_timestamps)
    rab_time_out = rab_time_out.t().view(num_layers, bs, infer_len, infer_len)

    return rab_time_out, bucket_timestamps
```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)
