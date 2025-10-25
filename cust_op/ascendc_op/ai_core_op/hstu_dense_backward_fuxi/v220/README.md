# HstuDenseBackward
本算子仅支持NPU调用

## 支持产品型号

Atlas A2 训练系列产品
产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/quickstart/quickstart/quickstart_18_0002.html)

## 功能描述

* 算子功能：推荐场景下，使用Hstu-fuxi融合算子实现推荐场景中的注意力机制
* 计算公式：
    $$
    qk=matmul(Q,K_{}^{T})
    $$
    $$
    score=Mask(Silu(qk))/S
    $$
    $$
    vGrad=matmul(score_{}^{T},G_{attn}) + matmul(Mask(biasTs)_{}^{T}, G_{biasTs}) + matmul(Mask(biasPos)_{}^{T}, G_{biasPos})
    $$
    $$
    qGrad=matmul(biasGrad,k)
    $$
    $$
    kGrad=matmul(biasGrad_{}^{T},q)
    $$
    $$
    tsGrad=Mask(matmul(G_{biasTs}, V_{}^{T}))
    $$
    $$
    posGrad=Mask(matmul(G_{biasPos}, V_{}^{T}))
    $$

其中Q,K,V是jagged格式

* jagged格式：s_b,N,D 3维数据格式 (稠密格式 为了节省显存)

jagged格式如下图所示：
![alt text](hstu_image-3.png)

## 实现原理

* 输入Q，K，V，Grad是jagged格式, 首先分别进行matmul计算
* mask和timestampBias/positionBias按照参数决定是否加入计算
* score和biasGrad的计算中，会除以序列长度S，jagged模式根据不同batch切换
* 最后score，biasGrad分别和Q/K/V做矩阵乘法得到输出梯度

## 算子输入与输出

| 名称 | 类型 | 数据类型 | 数据格式 | 备注 |
|----|----|----|----|----|
| grad | 输入| float32/float16/bfloat16 | jagged |
| q | 输入| float32/float16/bfloat16 | jagged |
| k | 输入| float32/float16/bfloat16 | jagged |
| v | 输入| float32/float16/bfloat16 | jagged |
| mask | 可选输入 | float32/float16/bfloat16 | B,N,S,S | S为模型最大的序列长度max_seq_len |
| bias_position | 可选输入 | float32/float16/bfloat16 | 1,S,S | S为模型最大的序列长度max_seq_len |
| bias_timestamp | 可选输入 | float32/float16/bfloat16 | B,S,S | S为模型最大的序列长度max_seq_len |
| grad_bias_position | 输入| float32/float16/bfloat16 | jagged |
| grad_bias_timestamp | 输入| float32/float16/bfloat16 | jagged |
| layout | 属性 | string | N/A | 当前仅支持"jagged"，“jagged”代表Q,K,V数据格式为s_b,N,D格式 |
| mask_type | 属性 | int | N/A | 0:使用内置下三角掩码 1:使用内置上三角掩码(未支持) 2:不使用mask(即使mask传值) 3:使用自定义mask(需要输入mask) |
| max_seq_len | 属性 | int | N/A | 表示模型最大序列长度 |
| silu_scale | 属性 | float | N/A | 支持用户传入自定义silu_scale, 不传入时默认值为1/max_seq_len|
| seq_offsets | 可选属性 | list[int64] | N/A | 表示每个序列的偏移，其中第一个序列的偏移一定是0，此选项只对jagged格式下生效，normal格式不生效。|
| q_grad | 输出 | float32/float16/bfloat16 | jagged |
| k_grad | 输出 | float32/float16/bfloat16 | jagged |
| v_grad | 输出 | float32/float16/bfloat16 | jagged |
| position_bias_grad | 输出 | float32/float16/bfloat16 | 1,S,S | S为变长序列中最大的序列长度 |
| timestamp_bias_grad | 输出 | float32/float16/bfloat16 | B,S,S | S为变长序列中最大的序列长度 |
| vbpos_grad | 输出 | float32/float16/bfloat16 | jagged | S为变长序列中最大的序列长度 |
| vbts_grad | 输出 | float32/float16/bfloat16 | jagged | S为变长序列中最大的序列长度 |

## 算子约束

* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* B: batch_size 表征批处理的大小，当前取值范围[1, 512]。
* S: seq_lens 表征序列长度，当前取值范围[1, 20480]。
* N：head_num 表征头个数，当前取值为[2,4,6,8]。
* D: head_dim 表征维度，当前取值范围范围[16, 512]，并且需要满足是16的倍数。
* 以上四个维度数值均不能为0，为0时算子输入为空数据，不会执行算子计算;并且其中B、N、S参数影响bias、mask占用显存大小，请根据实际内存合理设置参数大小。
* jagged模式下 S为所有序列中最大的序列长度，比如此时有两个序列，一个序列长度为256，另一个序列长度为512，则S为512。
* jagged模式，需要传递可选属性seq_offsets，比如当前有两个序列，一个序列长度为256,另一个序列长度为512，则seqp_offsets = [0, 256, 768]，伪代码如下:

```python
max_seq_len = 512
batch_size = 2
seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size)).astype(np.int64)

seq_offset = torch.concat((torch.zeros((1, ), dtype=torch.int64), \
        torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
```

## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)

### 执行样例

将RecSDK/examples/ops/torch/hstu_dense_backward_fuxi目录上传至目标环境，并进入目录；<br>
执行`pytest test_hstu_dense_backward_fuxi.py`<br>
该样例只能作为精度测试不能作为性能测试的基准；<br>
