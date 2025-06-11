# dense_to_jagged算子及样例说明

## dense_to_jagged算子文件结构

```shell
├── dense_to_jagged.json    # 算子原型配置
├── op_host    # dense_to_jagged算子Host侧实现
├── op_kernel  # dense_to_jagged算子Kernel侧实现
├── README.md  # dense_to_jagged算子说明文档
└── run.sh     # dense_to_jagged算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## dense_to_jagged算子使用

1. 上传dense_to_jagged文件夹到目标环境，并进入当前目录，执行指令对dense_to_jagged算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## dense_to_jagged算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的dense_to_jagged, 实现了将padded dense转为jagged tensor的功能
b) 算子参数说明：

* dense: 输入的padded dense tensor；
* offset: padded dense tensor中有效数据的偏移；
* jagged_dim0: jagged tensor的第一维长度；
* jagged_dense: 输出值;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：dense: float32/int64, offset: int64/int32；
* dense为3维tensor；
* offset为1维tensor，offset的长度必须为dense第一维长度加1，offset必须满足从0开始依次递增；
* jagged_dim0的值，需与offset最后一个值相等

## 算子逻辑
```
import numpy as np
def dense_to_jagged(dense, offset, jagged_dim0):
    jagged_dense = torch.zeros(jagged_dim0, dense.shape[2]， dtype=dense.dtype)

    for i in range(offset.shape[0] - 1):
        copyLen = offset[i + 1] - offset[i]
        jagged_dense[offset[i]:offset[i + 1], :] = dense[i][0:copyLen, :]

    return jagged_dense

```