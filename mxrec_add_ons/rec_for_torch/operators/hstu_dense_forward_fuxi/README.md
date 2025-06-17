# HstuForwardFuxi
本算子仅支持NPU调用

## 支持产品型号

Atlas A2 训练系列产品
Atlas 推理系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/quickstart/quickstart/quickstart_18_0002.html)

## 功能描述
* 算子功能: 推荐场景下，使用Hstu融合算子实现推荐场景中注意力机制
* 计算公式:
    $score = Mask(Silu(matmul(Q, K)) * siluScale)$
    $outputScore = matmul(score, V)$
    $outputTs = matmul(Mask(timestampBias), V)$
    $outputPos = matmul(Mask(positionBias), V)$
    $output = cat([outputScore, outputTs, outputPos], -1)$

其中Q,K,V 可以是normal格式，或者jagged格式。
* normal格式: B,S,N,D 4维数据格式
* jagged格式: s_b,N,D 3维数据格式 (稠密格式 为了节省显存)
normal 排布如下图所示:
![alt text](pic/hstu_image-2.png)
jagged 排布如下图所示:
![alt text](pic/hstu_image-3.png)


## 实现原理
![实现原理](pic/hstu_image.png)
* 输入Q，K是normal格式或者jagged格式, 首先进行matmul计算得到相关度分数
* 然后进行Silu最后除以系数S，如果是normal格式
S等于S轴的大小，如果是Jagged格式S是每个序列的长度，得到score。
* 最后score与V进行matmul得到最后的输出outputScore。
* 如果RAB参与计算，RAB分别做mask掩码计算后与V进行matmul得到bias的output。
* 不带RAB时，最终结果为outputScore；
* 带RAB时，最终结果为outputScore、outputBiasTs、outputBiasPos做cat操作，将最后一维整合得到最终结果

## 算子输入与输出
Atlas A2 训练系列产品
| 名称 | 类型 | 数据类型 | 数据格式 | 备注 |
|----|----|----|----|----|
| Q | 输入| float32/float16/bfloat16 | jagged |
| K | 输入| float32/float16/bfloat16 | jagged |
| V | 输入| float32/float16/bfloat16 | jagged |
| timestamp_bias | 可选输入 | float32/float16/bfloat16 | B,S,S | S为模型最大的序列长度max_seq_len，不使用时传入None |
| position_bias | 可选输入 | float32/float16/bfloat16 | 1,S,S | S为模型最大的序列长度max_seq_len，不使用时传入None |
| mask | 可选输入 | float32/float16/bfloat16 | B,N,S,S | S为模型最大的序列长度max_seq_len，不使用时传入None |
| maskType | 属性 | int | N/A | 0:使用内置倒三角mask 不需要传递mask输入 1:使用内置上三角mask 不需要传递mask输入当前暂不支持 2:不使用mask 3:使用用户自定义mask 此时mask输入需要用户定义并传入 |
| max_seq_len | 属性 | int | N/A | 表示模型最大序列长度 |
| siluScale | 属性 | float | N/A | 支持用户传入自定义siluScale, 不传入时默认值为1/max_seq_len|
| layout | 属性 | string | N/A | "normal"代表Q,K,V数据格式为B,S,N,D格式，“jagged”代表Q,K,V数据格式为s_b,N,D格式|
|seq_offsets| 可选属性 | list[int64] | N/A | 表示每个序列的偏移，其中第一个序列的偏移一定是0，此选项只对jagged格式下生效，normal格式不生效。
|output | 输出 | float32/float16/bfloat16 | jagged |

Atlas 推理系列产品
| 名称 | 类型 | 数据类型 | 数据格式 | 备注 |
|----|----|----|----|----|
| Q | 输入| float16 | normal |
| K | 输入| float16 | normal |
| V | 输入| float16 | normal |
| timestamp_bias | 可选输入 | float32/float16/bfloat16 | B,S,S | S为模型最大的序列长度max_seq_len，不使用时传入None |
| position_bias | 可选输入 | float32/float16/bfloat16 | 1,S,S | S为模型最大的序列长度max_seq_len，不使用时传入None |
| mask | 输入 |float16 | B,1,S,S | 掩码，当前仅支持normal格式，S为模型最大的序列长度max_seq_len |
| maskType | 属性 | int | N/A | 0:使用内置倒三角mask 不需要传递mask输入 1:使用内置上三角mask 不需要传递mask输入当前暂不支持 2:不使用mask 3:使用用户自定义mask 此时mask输入需要用户定义并传入，目前仅支持自定义mask|
| max_seq_len | 属性 | int | N/A | 表示模型最大序列长度 |
| siluScale | 属性 | float | N/A | 支持用户传入自定义siluScale，不传入时默认值为1/S， S为等长的序列长度|
| layout | 可选属性 | string | N/A |  当前仅支持normal，Q,K,V数据格式为B,S,N,D格式
| output | 输出 | float16 | normal| 输出与输入Q,K,V格式及数据类型保持一致

## 算子约束
Atlas A2 训练系列产品
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* B: batch_size 表征批处理的大小，当前取值范围[1, 512]。
* S: seq_lens 表征序列长度，当前取值范围[1, 20480]，并且需要满足是整数。
* N：head_num 表征头个数，当前取值为[2,4,6,8]。
* D: head_dim 表征维度，当前取值范围范围[16, 512]，并且需要满足是16的倍数。
* shape 信息需要传入正整数，如果是非正整数数据类型，会按照python的转换规则进行转换，比如当batch_size为布尔类型时，会按照0和1进行处理。
* 以上四个维度数值均不能为0，为0时算子输入为空数据，不会执行算子计算;并且其中B、N、S参数影响bias、mask占用显存大小，请根据实际内存合理设置参数大小。
* bias的shape中S为所有序列中最大的序列长度，比如此时有两个序列，一个序列长度为256，另一个序列长度为512，则S为512。
* 需要传递可选属性seq_offsets，比如当前有两个序列，一个序列长度为256,另一个序列长度为512，则seqp_offsets = [0, 256, 768]，伪代码如下:
```python
max_seq_len = 512
batch_size = 2
seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

seq_offset = torch.concat((torch.zeros((1, ), dtype=torch.int64), \
        torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
```
Atlas 推理系列产品
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* B: batch_size 表征批处理的大小，当前取值范围[1, 10]。
* S: seq_lens 表征序列长度，当前取值范围[64, 20480]，并且需要满足是64的倍数。
* N：head_num 表征头个数，当前取值为[2, 4, 6, 8]。
* D: head_dim 表征维度，当前取值范围范围[16, 128]，并且需要满足是16的倍数。
* shape 信息需要传入正整数，如果是非正整数数据类型，会按照python的转换规则进行转换，比如当batch_size为布尔类型时，会按照0和1进行处理。
* 以上四个维度数值均不能为0，为0时算子输入为空数据，不会执行算子计算;并且其中B、N、S参数影响bias、mask占用显存大小，请根据实际内存合理设置参数大小。

## 使用方式

### 下载软件包并解压
tar -zxvf Ascend-mindxsdk-mxrec-add-ons-linux-aarch64.tar.gz
   
### 部署安装算子
进入解压后的mxrec_ops目录
执行./mxrec_opps_hstu_dense_forward_fuxi.run 完成算子安装部署

### 编译torch适配层SO
进入解压后的torch_plugin/torch_library/2.6.0/hstu_dense_forward_fuxi目录
执行 build build_ops.sh命令完成torch适配层编译

### 执行样例
进入解压后的torch_plugin/torch_demo/hstu_dense_forward_fuxi目录
执行pytest test_hstu_forward_jagged_fuxi.py
该样例只能作为精度测试不能作为性能测试的基准