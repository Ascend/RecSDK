# hstu_dense_forward算子及样例说明
本算子仅支持NPU调用

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
* k: shape[batch_size, seq_len, num_head, d_qk]；
* v: shape[batch_size, seq_len, num_head, d_qk]；
* causal: 是否开启causal_mask;
* attn_bias: qk^T后加的bias参数， shape[batch_size, num_head, seq_len, seq_len];
* silu_scale: silu的系数;


c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC3及之后版本;
* 支持的输入数据类型：q、k、v支持fp32、fp16、bf16;
* seq_len须为256的倍数，范围：[256,4096];
* d_qk须为16的倍数， 范围：[16,512]


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