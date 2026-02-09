# disetangle_attention算子及样例说明
本算子仅支持NPU调用

## disetangle_attention算子文件结构

```shell
├── disetangle_attention.json    # 算子原型配置
├── op_host    # disetangle_attention算子Host侧实现
├── op_kernel  # disetangle_attention算子Kernel侧实现
├── README.md  # disetangle_attention算子说明文档
└── run.sh     # disetangle_attention算子安装脚本
```

## disetangle_attention算子介绍

1. 算子分析

* a) 算子的主要功能是实现deberta模型中的disetangle_attention，解耦注意力的功能
* b) 算子参数说明：

| 名称            | 输入/输出 | 数据类型 | 数据格式     | 备注               |
| --------------- | --------- | -------- | ------------ | ------------------ |
| query_layer     | 输入      | fp16     | (b,n,s,d)    |                    |
| key_layer       | 输入      | fp16     | (b,n,s,d)    |                    |
| value_layer     | 输入      | fp16     | (b,n,s,d)    |                    |
| pos_key_layer   | 输入      | fp16     | (2s, n, d)   |                    |
| pos_query_layer | 输入      | fp16     | (2s, n, d)   |                    |
| relative_pos    | 输入      | int64    | (1,1,s,s)    |                    |
| atten_mask      | 输入      | fp16     | (b, 1, s, s) |                    |
| pos_attr_type   | 属性      | str      | N/A          | 'c2p', 'p2c', 'c2p\|p2c' |
| score_scale     | 属性      | float    | N/A          |                    |
| atten_outputs     | 输出      | fp16     | (b,n,s,d)    |                    |
| atten_probs      | 输出      | fp16     | (b,n,s,s)    |                    |
| atten_weights    | 输出      | fp16     | (b,n,s,s)    |                    |

ps: 

* 如果pos_attr 传入"c2p"只会计算content2postion编码偏置
* 如果pos_attr 传入"p2c"只会计算position2context编码偏置
* 如果pos_attr 传入"c2p|p2c" 则都会计算并将上述两个偏置累加后输出

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* b: 取值范围[0, 100]
* n: 取值范围[0, 64]
* s: 取值为256
* d: 取值范围[1, 512]
* 数据类型: fp16

## 算子逻辑
```mermaid
graph TD
    %% ====== 内容到位置路径（c2p） ======
    InputQuery["输入 query_layer<br/>Shape: BxH, Q, D"] --> MatMul1["MatMul"]
    InputPosKey["输入 pos_key_layer<br/>Shape: BxH, P, D"] --> Transpose["Transpose<br/>permute: -1,-2"]
    Transpose -->|"pos_key_layer_T<br/>Shape: BxH, D, P"| MatMul1
    MatMul1 -->|"c2p_att_raw<br/>Shape: BxH, Q, P"| Scale1["Scale<br/>factor: 1/&radic;(D×scale_factor)"]
    Scale1 -->|"c2p_att<br/>Shape: BxH, Q, P"| Gather1["Gather"]
    
    InputRelPos["输入 relative_pos<br/>Shape: B, H, Q, K"] --> Clamp["Clamp<br/>relative_pos + att_span → 0:2×att_span-1"]
    Clamp -->|"c2p_pos_raw<br/>Shape: B, H, Q, K"| Reshape["Reshape<br/>view: BxH, Q, K"]
    Reshape -->|"c2p_pos<br/>Shape: BxH, Q, K"| Gather1
    Gather1 -->|"c2p_output<br/>Shape: BxH, Q, K"| Add["Add"]
    
    %% ====== 位置到内容路径（p2c） ======
    InputPosQuery["输入 pos_query_layer<br/>Shape: BxH, P, D"] --> MatMul2["MatMul"]
    InputKey["输入 key_layer<br/>Shape: BxH, K, D"] --> Transpose2["Transpose<br/>permute: -1,-2"]
    Transpose2 -->|"key_layer_T<br/>Shape: BxH, D, K"| MatMul2
    MatMul2 -->|"p2c_att_raw<br/>Shape: BxH, P, K"| Scale2["Scale<br/>factor: 1/&radic;(D×scale_factor)"]
    Scale2 -->|"p2c_att<br/>Shape: BxH, P, K"| Gather2["Gather<br/>dim=-2"]
    Reshape -->|"c2p_pos<br/>Shape: BxH, Q, K"| Gather2
    Gather2 -->|"p2c_output<br/>Shape: BxH, Q, K"| Add
    
    %% ====== 结果输出 ======
    Add -->|"score<br/>Shape: BxH, Q, K"| Output["输出 disentangled_bias"]

```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)