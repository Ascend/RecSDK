# hstu_dense_backward算子及样例说明
本算子仅支持NPU调用

## hstu_dense_backward算子文件结构

```shell
├── hstu_dense_backward.json    # 算子原型配置
├── op_host    # hstu_dense_backward算子Host侧实现
├── op_kernel  # hstu_dense_backward算子Kernel侧实现
├── README.md  # hstu_dense_backward算子说明文档
└── run.sh     # hstu_dense_backward算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## hstu_dense_backward算子使用

1. 上传hstu_dense_backward文件夹到目标环境，并进入当前目录，执行指令对hstu_dense_backward算子进行编译和部署

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

## hstu_dense_backward算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的hstu_dense_backward, 实现embedding bag的查询功能
b) 算子参数说明：
* grad;
* q: shape[batch_size, seq_len, num_head, d_qk]；
* k:  shape[batch_size, seq_len, num_head, d_qk]；
* v: shape[batch_size, seq_len, num_head, d_qk]；
* attn_bias: qk^T后加的bias参数， shape[batch_size,num_head, seq_len, seq_len];

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC3及之后版本；
* 支持的输入数据类型：q、k、v支持fp32、fp16、bfp16;
* seq_len须为256的倍数，范围：[256,4096];
* d_qk须为16的倍数， 范围：[16,512]


## 算子逻辑
```
q_trans = q.permute(0, 2, 1, 3)
k_trans = k.permute(0, 2, 1, 3)
v_trans = v.permute(0, 2, 1, 3)
g_trans = grad.permute(0, 2, 1, 3)
qk_result = torch.matmul(q_trans, k_trans.permute(0, 1, 3, 2)) 
gv_result = torch.matmul(g_trans, v_trans.permute(0, 1, 3, 2))

qk_add_atten_bias = qk_result + attn_bias
attn_score_forward = F.silu(qk_add_atten_bias) / max_seq_len
attn_score_forward = attn_score_forward * invalid_attn_mask.unsqueeze(0).unsqueeze(0)

score_grad = gv_result / max_seq_len * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
qk_attn_grad = (F.sigmoid(qk_add_atten_bias)*(1+qk_add_atten_bias*(1-F.sigmoid(qk_add_atten_bias)))) * score_grad
rel_attn_bias_grad = qk_attn_grad.sum(1, keepdim=True)

# # V grad
v_grad = torch.matmul(attn_score_forward.permute(0, 1, 3, 2), g_trans)

# # Q K grad
q_grad = torch.matmul(qk_attn_grad, k_trans)
k_grad = torch.matmul(qk_attn_grad.permute(0, 1, 3, 2), q_trans)
```