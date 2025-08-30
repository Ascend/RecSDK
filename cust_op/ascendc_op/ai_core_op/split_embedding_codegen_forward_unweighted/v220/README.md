# split_embedding_codegen_forward_unweighted算子及样例说明
本算子仅支持NPU调用

## split_embedding_codegen_forward_unweighted算子文件结构

```shell
├── split_embedding_codegen_forward_unweighted.json    # 算子原型配置
├── op_host    # split_embedding_codegen_forward_unweighted算子Host侧实现
├── op_kernel  # split_embedding_codegen_forward_unweighted算子Kernel侧实现
├── README.md  # split_embedding_codegen_forward_unweighted算子说明文档
└── run.sh     # split_embedding_codegen_forward_unweighted算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## split_embedding_codegen_forward_unweighted算子使用

1. 上传split_embedding_codegen_forward_unweighted文件夹到目标环境，并进入当前目录，执行指令对split_embedding_codegen_forward_unweighted算子进行编译和部署

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

## split_embedding_codegen_forward_unweighted算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的split_embedding_codegen_forward_unweighted, 实现embedding bag的查询功能

b) 算子参数说明：

* dev_weights: 表的权重；
* uvm_weight: 预留参数不支持配置；
* lxu_cache_weight: 预留参数不支持配置；
* weights_pacements: 预留参数不支持配置;
* weights_offsets: 每张表的偏移量;
* D_offsets: 每张表embedding dim的累加和偏移量;
* indices: 查询表的索引;
* hash_indices: 稀疏表查表的索引，可选参数;
* offsets: 查表索引对应的偏移;
* lxu_cache_locations: 预留参数不支持配置;
* out: 查询的向量；
* total_D: 输出的embedding dim之和;
* max_D: 表中最大的embedding dim;;
* pooling_mode: pooling的方式Sum或者Mean或None;
* output_dtype: 预留参数不支持配置;
* is_experimental: 预留参数不支持配置;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
* 支持的输入数据类型：dev_weights为float32类型，weights_offsets、indices、hash_indices、offsets为int64，D_offsets为int32。
* 支持的输入shape： dev_weights的dim为所有表的[embed_dim * embed_size]之和，embed_dim长度需为8的倍数, pooling_mode为None时所有表的embed_dim需要保持一致。  
                   weights_offsets的dim为[ feat_cnt ]。  
                   D_offsets的dim为[ feat_cnt + 1 ]。  
                   offsets的dim为[batchsize * feat_cnt + 1]，请注意：offsets中的元素大小超过10000时，可能出现累加误差。  
                   indices的dim与offset最后一个元素大小相同。  

## 算子逻辑
```
# with bag sum or mean
def split_embedding_codegen_forword_unweighted(dev_weights, weights_offsets, D_offsets, indices, offsets, total_D, pool_mode):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0]-1) // feat_cnt
    results = np.zeros((batch_size, total_D)).astype(np.float32)
    for i in range(feat_cnt):
        embed_dim = D_offsets[i+1] - D_offsets[i]
        for b in range(batch_size):
            this_offset = offsets[i*batch_size+b]
            next_offset = offsets[i*batch_size+b+1]
            this_indics = indices[this_offset: next_offset]
            # sum
            if pool_mode == 0:
                seq_lens = 1
            # mean
            else:
                seq_lens = len(this_indics)
            for j in this_indics:
                this_embed_index = weights_offsets[i]+j*embed_dim
                this_embed = dev_weights[this_embed_index: this_embed_index+embed_dim]
                results[b, D_offsets[i]:D_offsets[i+1]] = results[b, D_offsets[i]:D_offsets[i+1]] + this_embed/seq_lens
    return result.astype(np.float32)

# with no bag
def split_embedding_nobag_codegen_forword_unweighted(dev_weights, weights_offsets, indices, offsets, total_D):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0]-1) // feat_cnt
    out_D0 = len(indices)
    out_D1 = total_D // feat_cnt  # EC模式下要求每张表的dim一致
    results = np.zeros((out_D0, out_D1)).astype(np.float32)
    result_indx = 0
    for i in range(len(offsets)-1):
        # 待查indics
        this_indice = indices[offsets[i]:offsets[i+1]]
        # 待查表
        weights_indx = i // batch_size
        for j in this_indice:
            this_embed_index = weights_offsets[weights_indx] + j * out_D1
            this_embed = dev_weights[this_embed_index: this_embed_index + out_D1]
            results[result_indx] = this_embed
            result_indx += 1
    return result.astype(np.float32)

```