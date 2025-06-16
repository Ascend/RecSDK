# permute2d_sparse_data算子及样例说明

## permute2d_sparse_data算子文件结构

```shell
├── permute2d_sparse_data.json    # 算子原型配置
├── op_host    # permute2d_sparse_data算子Host侧实现
├── op_kernel  # permute2d_sparse_data算子Kernel侧实现
├── README.md  # permute2d_sparse_data算子说明文档
└── run.sh     # permute2d_sparse_data算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## permute2d_sparse_data算子使用

1. 上传permute2d_sparse_data文件夹到目标环境，并进入当前目录，执行指令对permute2d_sparse_data算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## permute2d_sparse_data算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的permute2d_sparse_data, 实现了对二维稀疏数据进行重排。
b) 算子参数说明：

* permute: 重排的顺序参数tensor；
* lengths: 待重排长度参数；
* values: 待重排值参数；
* weights: 暂不支持使用
* permute_sum: 暂不支持使用
* permuted_lengths: 输出， 重排后长度tensor;
* permuted_values: 输出，重排后值tensor;
* permuted_weights: 输出， 暂不支持

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：permute: int32, lengths: int64/int32, values: int64/int32/float；
* permute为1维tensor，lengths为二维tensor，且permute的第一维长度与lengths的第一维长度相等；
* values长度为lengths中所有数据长度之和
* 算子参数均会在NPU显存中存放，请根据显存大小合理设置参数长度。

## 算子逻辑
```
import numpy as np
def permute2d_sparse_data(permute, lengths, values):
    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, values)
    )

    return permuted_lengths, permuted_values

```