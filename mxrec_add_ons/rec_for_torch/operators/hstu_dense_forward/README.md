# hstu_dense_forward算子及样例说明

## hstu_dense_forward算子文件结构

```shell
├── hstu_dense_forward.json    # 算子原型配置
├── op_host    # hstu_dense_forward算子Host侧实现
├── op_kernel  # hstu_dense_forward算子Kernel侧实现
├── README.md  # hstu_dense_forward算子说明文档
└── run.sh     # hstu_dense_forward算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## hstu_dense_forward算子使用

1. 上传hstu_dense_forward文件夹到目标环境，并进入当前目录，执行指令对hstu_dense_forward算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## hstu_dense_forward算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的hstu_dense_forward
b) 算子参数说明：

* q: shape[batch_size, seq_len, num_head, d_qk]；
* k:  shape[batch_size, seq_len, num_head, d_qk]；
* v: shape[batch_size, seq_len, num_head, d_qk]；
* causal: 是否开启causal_mask;
* attn_bias: qk^T后加的bias参数， shape[batch_size,num_head, seq_len, seq_len];
* silu_scale: silu的系数;


c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC3及之后版本;
* 支持的输入数据类型：q、k、v支持fp32、fp16、bfp16;
* seq_len须为256的倍数，范围：[256,4096];
* d_qk须为32的倍数， 范围：[32,512]


## 算子逻辑
```
def hstu_dense_forward(q_np, k_np, v_np, rel_attn_bias_np, invalid_attn_mask_np):
    q = torch.nn.Parameter(torch.Tensor(q_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu(), 
        requires_grad=True)
    k = torch.nn.Parameter(torch.Tensor(k_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu(), 
        requires_grad=True)
    v = torch.nn.Parameter(torch.Tensor(v_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu(), 
        requires_grad=True)
    real_attn_bias = torch.nn.Parameter(torch.Tensor(rel_attn_bias_np).to(dataType).npu(), requires_grad=True)   
    invalid_attn_mask = torch.Tensor(invalid_attn_mask_np).to(dataType).npu()
     
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        q,
        k,
    )
    qk_attn = qk_attn + rel_attn_bias

    qk_attn = F.silu(qk_attn) / max_seq_len
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            v
        ).reshape(batch_size, seq_len, num_heads * attention_dim)
    b = attn_output.sum()
    b.backward()
    return npu2cpu(attn_output), npu2cpu(q.grad), npu2cpu(k.grad), npu2cpu(v.grad), 
            npu2cpu(real_attn_bias.grad), invalid_attn_mask

```

# HSTU Paged

## 概述

`hstu_paged` 是一个高性能的分页注意力机制实现。该接口实现了基于分页内存管理的hstu Attention计算。

## 参数说明

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `q` | Tensor| | Query张量，3D张量 [batch_size, seq_len, head_dim] |
| `k` | Tensor| | Key张量，3D张量 [batch_size, seq_len, head_dim] |
| `v` | Tensor| | Value张量，3D张量 [batch_size, seq_len, head_dim] |
| `kv_cache` | Tensor | None | KV缓存张量，用于存储历史Key-Value对 |
| `mask` | Tensor | None | 注意力掩码张量 |
| `attn_bias` | Tensor | None | 注意力偏置张量 |
| `mask_type` | int | 0 | 掩码类型：0=下三角，1=上三角，3=自定义 |
| `max_seq_len` | int64 | 0 | 最大序列长度，范围[1, 20480] |
| `max_seq_len_k` | int64 | 0 | Key最大序列长度，范围[1, 20480] |
| `silu_scale` | float | 0.0 | SiLU激活函数的缩放因子 |
| `seq_offset` | Tensor | None | 序列偏移量张量 |
| `seq_offset_k` | Tensor | None | Key序列偏移量张量 |
| `seq_offset_t` | Tensor | None | 目标序列偏移量张量 |
| `page_offsets` | Tensor | None | 页面偏移量张量 |
| `page_ids` | Tensor | None | 页面ID张量 |
| `last_page_len` | Tensor | None | 最后一页长度张量 |
| `num_target` | Tensor | None | 目标数量张量 |
| `target_group_size` | int | 0 | 目标组大小，当num_target定义时必须>0 |
| `alpha` | float| 1.0 | Alpha缩放参数 |


## 约束条件
* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC3及之后版本;
* 支持的输入数据类型：q、k、v、kv_cache支持fp32、fp16、bfp16;
* page_size支持: [32, 64, 128, 256]
* target_group_size：当num_target定义时必须>0