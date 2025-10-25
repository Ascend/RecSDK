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

## jagged_to_padded_dense算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的jagged_to_padded_dense, 实现了将jagged tensor转为padded dense的功能
b) 算子参数说明：

* values: 输入的jagged tensor；
* offsets: jagged tensor对应的位置；
* max_length: int数组，元素值为padded dense的第二维长度；
* padding_value: 填充值;
* out: 输出值;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：values float32/int64, offset int64/int32；
* values为2维tensor，offset为1维tensor，max_length的元素值大于0
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

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)