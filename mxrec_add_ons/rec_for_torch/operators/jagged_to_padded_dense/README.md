# jagged_to_padded_dense算子及样例说明
本算子仅支持NPU调用

## jagged_to_padded_dense算子文件结构

```shell
├── jagged_to_padded_dense.json    # 算子原型配置
├── op_host    # jagged_to_padded_dense算子Host侧实现
├── op_kernel  # jagged_to_padded_dense算子Kernel侧实现
├── README.md  # jagged_to_padded_dense算子说明文档
└── run.sh     # jagged_to_padded_dense算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## jagged_to_padded_dense算子使用

1. 上传jagged_to_padded_dense文件夹到目标环境，并进入当前目录，执行指令对jagged_to_padded_dense算子进行编译和部署

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

## jagged_to_padded_dense算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的jagged_to_padded_dense, 实现了将jagged tensor转为padded dense的功能
b) 算子参数说明：

* values: 输入的jagged tensor；
* offsets: jagged tensor对应的位置；
* max_length: padded dense的第二维长度；
* padding_value: 填充值;
* out: 输出值;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：values float32/int64, offset int64/int32；
* values为3维tensor，offset为1维tensor，max_length大于0
* offset必须满足从0开始依次递增
* 算子参数均会在NPU显存中存放，请根据显存大小合理设置参数长度。

## 算子逻辑
```
import numpy as np
def jagged_to_padded_dense(value, offsets, max_length, padding_value):
    out = np.full((offsets.shape[0]-1, max_length[0], value.shape[1]), padding_value).astype(np.float32)
    for i in range(1, offsets.shape[0]):
        copyLen = offsets[i] - offsets[i-1]
        out[i-1][0:copyLen, :] = value[offsets[i-1]: offsets[i]]
    return out.astype(np.float32)

value = np.arange(0, 13400*50).reshape(13400, 50).astype(np.float32)
offsets = np.random.randint(0, 10, (128)).astype(np.int64)
offsets = np.cumsum(offsets)
offsets = np.insert(offsets, 0, 0)

result = jagged_to_padded_dense(value=value, offsets=offsets, max_length=[211], padding_value=0)

```