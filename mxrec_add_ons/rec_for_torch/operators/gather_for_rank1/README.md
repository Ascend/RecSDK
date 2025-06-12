# gather_for_rank1算子及样例说明

## gather_for_rank1算子文件结构

```shell
├── gather_for_rank1.json    # 算子原型配置
├── op_host    # gather_for_rank1算子Host侧实现
├── op_kernel  # gather_for_rank1算子Kernel侧实现
├── README.md  # gather_for_rank1算子说明文档
└── run.sh     # gather_for_rank1算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## gather_for_rank1算子使用

1. 上传gather_for_rank1文件夹到目标环境，并进入当前目录，执行指令对gather_for_rank1算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## gather_for_rank1算子介绍

1. 算子分析

a) 算子的主要功能是实现x的shape为1的index_select
b) 算子参数说明：
* x: shape[embed_dim]；
* index:  shape[index_num]；


c) 算子约束说明：

* 支持的型号：Atlas A2系列产品和Atlas 推理系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：在Atlas A2系列产品中：x支持fp32、fp16, index支持int64_t;
*                   在Atlas 推理系列产品中：x支持fp32、fp16, index支持int32_t;
* x范围：[1,20480];
* index范围:[1,]


## 算子逻辑
```
xDim0 = 129
indexDim0 = 128*211*211
x = torch.randn(xDim0).to(torch.float32)
index = torch.randint(0, xDim0, (indexDim0, )).to(torch.int64)
y = torch.index_select(x, dim=0, index=index)
```