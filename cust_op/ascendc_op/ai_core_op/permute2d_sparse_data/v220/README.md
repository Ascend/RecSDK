# permute2d_sparse_data算子及样例说明
本算子仅支持NPU调用

## permute2d_sparse_data算子文件结构

```shell
├── permute2d_sparse_data.json    # 算子原型配置
├── op_host    # permute2d_sparse_data算子Host侧实现
├── op_kernel  # permute2d_sparse_data算子Kernel侧实现
├── README.md  # permute2d_sparse_data算子说明文档
└── run.sh     # permute2d_sparse_data算子安装脚本
```

## permute2d_sparse_data算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的permute2d_sparse_data, 实现了对二维稀疏数据进行重排。
b) 算子参数说明：

* permute: 重排的顺序参数tensor;
* lengths: 待重排长度参数;
* values: 待重排序的1D-tensor;
* weights: 可选入参，待重排序的1D-tensor。与values执行相同操作;
* permuted_lengths_sum: 可选入参，values/weights有效长度;
* permuted_lengths: 输出， 重排后长度tensor;
* permuted_values: 输出，重排后的values;
* permuted_weights: 输出，重排后的weights;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本;
* 支持的输入数据类型：
  * permute: int32;
  * lengths: int64/int32;
  * values: int64/int32/fp32;
  * weights: fp32;
  * permuted_lengths_sum: int(标量);
* permute为1维tensor，lengths为二维tensor，permute中的每个值均满足: >= 0 且 < `lengths.shape[0]`;
* 指定permuted_lengths_sum时，permuted_values/permuted_weights长度为permuted_lengths_sum，请用户自行保证数值正确;
* 未指定permuted_lengths_sum时，算子将计算得到permuted_lengths_sum;
* weights和values长度相同，均等于`lengths.sum()`;
* 算子参数均会在NPU显存中存放，请根据显存大小合理设置参数长度。

## 算子逻辑
```
import torch
import fbgemm_gpu
def permute2d_sparse_data(permute, lengths, values, weights, permuted_lengths_sum):
    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, values, weights, permuted_lengths_sum)
    )

    return permuted_lengths, permuted_values, permuted_weights

```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)