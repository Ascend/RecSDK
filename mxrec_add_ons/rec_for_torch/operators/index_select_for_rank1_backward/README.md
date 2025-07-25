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

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## index_select_for_rank1_backward融合算子使用

1. 上传index_select_for_rank1_backward文件夹到目标环境，并进入当前目录，执行指令对index_select_for_rank1_backward算子进行编译和部署

默认编译安装Atlas A2训练系列产品AI Core类型：
```shell
bash run.sh
```

指定 AI Core 类型编译：

```shell
bash run.sh ai_core-<soc_version>
```
> AI处理器的型号<soc_version>请通过如下方式获取:
> - 在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，获取`Chip Name`信息。实际配置值为AscendChip Name，例如`Chip Name`取值为`xxxyy`，实际配置值为`Ascendxxxyy`。
>
> 基于同系列的AI处理器型号创建的算子工程，其基础功能（基于该工程进行算子开发、编译和部署）通用。

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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