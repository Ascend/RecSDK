# relative_attn_bias_time_backward优化器融合算子及样例说明
本算子仅支持NPU调用

## relative_attn_bias_time_backward融合算子文件结构

```shell
├── relative_attn_bias_time_backward.json    # 算子原型配置
├── op_host    # relative_attn_bias_time_backward融合算子Host侧实现
├── op_kernel  # relative_attn_bias_time_backward融合算子Kernel侧实现
├── README.md  # relative_attn_bias_time_backward融合算子说明文档
└── run.sh     # relative_attn_bias_time_backward融合算子安装脚本
```

## relative_attn_bias_time_backward融合算子介绍

1. 算子分析

a) 算子参数说明：

| 算子参数                    | 输入/输出 | dtype     | shape            |
|-------------------------|-------|-----------|------------------|
| rab_time_grad           | 输入    | FP16,FP32 | (n, b, 2s, 2s)   |
| bucket_timestamps       | 输入    | int32     | (2s, 2s)         |
| num_buckets             | 输入    | int       |                  |
| timestamps_weights_grad | 输出    | FP16,FP32 | (n, num_buckets) |


b) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；

## 算子逻辑

```python
import torch
NUM_BUCKETS = 128 + 1

def rab_time_backward_golden(rab_time_grad: torch.Tensor, bucket_timestamps: torch.Tensor):
    num_layers, b, s, _ = rab_time_grad.shape
    tsw_grad = torch.zeros(num_layers, NUM_BUCKETS, dtype=torch.float32).to(rab_time_grad.device)

    bucket_timestamps_expand = (bucket_timestamps.reshape(b, s // 2, 1, s // 2, 1)
                                .repeat(1, 1, 2, 1, 2)
                                .reshape(b, s, s)
                                .to(torch.int64))
    for n, grad in enumerate(rab_time_grad.to(torch.float32)):
        tsw_grad[n], _ = torch.ops.mxrec.index_select_for_rank1_backward(grad.view(-1),
                                                                         tsw_grad[n],
                                                                         bucket_timestamps_expand.view(-1))
    return tsw_grad
```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)