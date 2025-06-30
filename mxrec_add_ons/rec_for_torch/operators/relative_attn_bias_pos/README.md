# relative_attn_bias_pos优化器融合算子及样例说明
本算子仅支持NPU调用

## relative_attn_bias_pos融合算子文件结构

```shell
├── relative_attn_bias_pos.json    # 算子原型配置
├── op_host    # relative_attn_bias_pos融合算子Host侧实现
├── op_kernel  # relative_attn_bias_pos融合算子Kernel侧实现
├── README.md  # relative_attn_bias_pos融合算子说明文档
└── run.sh     # relative_attn_bias_pos融合算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## relative_attn_bias_pos融合算子使用

1. 上传relative_attn_bias_pos文件夹到目标环境，并进入当前目录，执行指令对relative_attn_bias_pos算子进行编译和部署

默认编译安装Atlas A2训练系列产品AI Core类型：
```shell
bash run.sh
```

指定 AI Core 类型编译：

```shell
bash run.sh ai_core-<soc_version>
```
> AI处理器的型号<soc_version>请通过如下方式获取:
> - 在安装晟腾AI处理器的服务器执行`npu-smi info`命令进行查询，获取`Chip Name`信息。实际配置值为AscendChip Name，例如`Chip Name`取值为`xxxyy`，实际配置值为`Ascendxxxyy`。
>
> 基于同系列的AI处理器型号创建的算子工程，其基础功能（基于该工程进行算子开发、编译和部署）通用。

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## relative_attn_bias_pos融合算子介绍

1. 算子分析

a) 算子参数说明：

| 算子参数              | 输入/输出 | dtype     | shape       |
|-------------------|-------|-----------|-------------|
| rel_pos_bias      | 输入    | FP16,FP32 | (2s, 2s)    |
| identity          | 输入    | FP16,FP32 | (2s, 2s)    |
| past_valid_lens   | 输入    | List[int] | (b,)        |
| rab_pos           | 输出    | FP16,FP32 | (b, 2s, 2s) |


b) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；

## 算子逻辑

```python
import torch


def rab_pos_golden(rel_pos_bias: torch.Tensor, identity: torch.Tensor, past_valid_lens: torch.Tensor) -> torch.Tensor:
    """
    past_len = 1 ~ 4000
    candidate_len = 256 ~ 600
    bs = 1 ~ 10

    :param rel_pos_bias: [past_len * 2 + candidate_len][past_len * 2 + candidate_len]
    :param identity: [past_len * 2 + candidate_len][past_len * 2 + candidate_len]
    :param past_valid_lens: [bs]
    :return: [bs][1][past_len * 2 + candidate_len + 2][past_len * 2 + candidate_len + 2]
    """
    bs = past_valid_lens.shape[0]
    rel_pos_bias_list = rel_pos_bias[:].unsqueeze(0).repeat(bs, 1, 1)
    for i, valid_len in enumerate(past_valid_lens):
        rel_pos_bias_list[i, valid_len:, :] = rel_pos_bias[valid_len, :]

    rel_pos_bias_list = rel_pos_bias_list * (1 - identity) + identity * rel_pos_bias_list[0, 0, 0]
    rel_pos_bias_list = rel_pos_bias_list[:, :identity.shape[0], :identity.shape[0]]
    return rel_pos_bias_list
```
