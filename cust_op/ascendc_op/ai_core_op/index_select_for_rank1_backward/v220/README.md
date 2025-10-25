# index_select_for_rank1_backward优化器融合算子及样例说明
本算子仅支持NPU调用

## index_select_for_rank1_backward融合算子文件结构

```shell
├── index_select_for_rank1_backward.json    # 算子原型配置
├── op_host    # index_select_for_rank1_backward融合算子Host侧实现
├── op_kernel  # index_select_for_rank1_backward融合算子Kernel侧实现
├── README.md  # index_select_for_rank1_backward融合算子说明文档
└── run.sh     # index_select_for_rank1_backward融合算子安装脚本
```

## index_select_for_rank1_backward融合算子介绍

1. 算子分析

a) 算子的主要功能是实现index_select的反向函数, 聚合所有的grad

b) 算子参数说明：

* grad_y: index select的反向grad；
* x: index select查询的tensor；
* index: index select查询的index；
* grad_x: x的梯度;
* grad_index: index的梯度;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：grad_y float32, x float32, index int64/int32；
* values、x、index都为1维
* index内的值不得超过x第0长度

## 算子逻辑
```
import numpy as np

def index_select_for_rank1_backward(grad, x, index):
    grad_x = np.zeros(x.shape).astype(np.float32)
    grad_index = np.zeros(index.shape)
    for i in range(x.shape[0]):
        out = grad[index == i]
        out = out.sum()
        grad_x[i] = out
    return grad_x.astype(np.float32), grad_index


grad = np.ones([128, 211, 211]).astype(np.float32)
x = np.ones([129]).astype(np.float32)
index = np.arange(128)[:, np.newaxis, np.newaxis].repeat(211, axis=1).repeat(211, axis=2) 

grad_x, grad_index = index_select_for_rank1_backward(grad, x, index)

```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)