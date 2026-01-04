**说明**

本算子仅支持 NPU 调用。

# 产品支持情况
| 硬件型号 | 是否支持 |
| --- | --- |
| Atlas A5 训练系列产品 | 是 |

# 目录结构
```shell
-- int_nbit_split_embedding_codegen_lookup_function
   |-- c310
      |-- op_host                                   # 算子 host 侧实现
      |-- op_kernel                                 # 算子 kernel 侧实现
      |-- int_nbit_split_embedding_codegen_lookup_function.json
      |-- README.md
      |-- run.sh
```

# 功能简介
实现 fbgemm `int_nbit_split_embedding_codegen_lookup_function` 的前向查表，支持 bag（SUM/MEAN）与 nobag、FP8 量化权重、FP32/FP16/BF16/INT8 输出。

# 实现原理概述 (伪代码)
```python
def int_nbit_split_embedding(dev_weights, weights_offsets, weights_tys,
                             D_offsets, indices, offsets,
                             pooling_mode, indice_weights=None,
                             output_dtype=SparseType.FP32):
    feat_cnt = len(weights_offsets)
    if pooling_mode == PoolingMode.NONE:  # nobag
        out_dim1 = max_row_dim(weights_tys)
        outputs = []
        for t in range(feat_cnt):
            for idx in indices[offsets[t]: offsets[t+1]]:
                row = load_quant_row(dev_weights, weights_offsets[t], idx)
                vec = dequantize_fp8(row)
                outputs.append(cast(vec, output_dtype))
        return stack(outputs)
    else:  # bag SUM / MEAN
        batch = (len(offsets) - 1) // feat_cnt
        result = zeros((batch, D_offsets[-1]), dtype=output_dtype)
        for t in range(feat_cnt):
            dim_start, dim_end = D_offsets[t], D_offsets[t+1]
            for b in range(batch):
                start = offsets[t * batch + b]
                end = offsets[t * batch + b + 1]
                acc = zeros(dim_end - dim_start, dtype=float)
                for k, idx in enumerate(indices[start:end]):
                    row = load_quant_row(dev_weights, weights_offsets[t], idx)
                    vec = dequantize_fp8(row)
                    weight = indice_weights[start + k] if indice_weights is not None else 1.0
                    acc += vec * weight
                if pooling_mode == PoolingMode.MEAN and (end - start) > 0:
                    acc /= (end - start)
                result[b, dim_start:dim_end] = cast(acc, output_dtype)
        return result
```

# 输入输出概要
| 名称 | 输入/输出 | 数据类型 | 数据格式                                           | 范围 | 说明                                              |
| --- |-------| --- |------------------------------------------------| --- |-------------------------------------------------|
| dev_weights | 输入    | uint8 | [total_weight_bytes]                           | NA | 多表合并后的量化权重缓冲区                                   |
| uvm_weights | 输入    | uint8 | NA                                             | NA | 保留参数                                            |
| lxu_cache_weights | 输入    | uint8 | NA                                             | NA | 保留参数                                            |
| weights_placements | 输入    | int32 | [T]                                            | PlacementType | 每张表的放置策略（DEVICE/HOST/MANAGED/…）                 |
| weights_offsets | 输入    | int64 | [T]                                            | 单调递增 | 每张表在 `dev_weights` 中的起始偏移                       |
| weights_tys | 输入    | uint8 | [T]                                            | SparseType | 每张表的量化类型，暂时只支持FP8（FP8/INT8/FP16/FP32/INT4/INT2） |
| D_offsets | 输入    | int32 | [T+1]                                          | 单调递增 | embedding 维度前缀和，用于定位各表输出区间                      |
| indices | 输入    | int32/64 | [N]                                            | offsets[-1] | 查表索引（nobag 在适配层转成 int32）                        |
| offsets | 输入    | 与 indices 同型 | [T*B+1]                                        | 单调递增 | bag 的起止偏移                                       |
| lxu_cache_locations | 输入    | int32 | NA                                             | NA | 保留参数                                            |
| offset_per_key | 输入    | int32 | [T+1]                                          | 单调递增 | 每张表在 offsets 中的起点（仅 nobag 模式使用）                 |
| indice_weights | 输入    | float | [N]                                            | NA | weighted bag 的 per-sample 权重，非加权时传空             |
| total_D | 属性    | int64 | NA                                             | NA | 所有表 embedding 维度之和（bag 输出宽度）                    |
| max_D | 属性    | int64 | NA                                             | NA | 各表 embedding 维度的最大值（nobag 输出宽度）                 |
| max_int2_D | 属性    | int64 | NA                                             | NA | INT2 权重量化的最大维度                                  |
| max_int4_D | 属性    | int64 | NA                                             | NA | INT4 权重量化的最大维度                                  |
| max_int8_D | 属性    | int64 | NA                                             | NA | INT8 权重量化的最大维度                                  |
| max_float16_D | 属性    | int64 | NA                                             | NA | FP16 权重量化的最大维度                                  |
| max_float32_D | 属性    | int64 | NA                                             | NA | FP32 权重（或输出）最大维度                                |
| max_float8_D | 属性    | int64 | NA                                             | NA | FP8 权重量化的最大维度                                   |
| pooling_mode | 属性    | int64 | NA                                             | {0,1,2} | 0=SUM，1=MEAN，2=NONE（nobag）                      |
| output_dtype | 属性    | int64 | NA                                             | SparseType | 输出数据类型（FP32/FP16/BF16/INT8）                     |
| row_alignment | 属性    | int64 | NA                                             | NA | 行级对齐要求（默认为 16）                                  |
| fp8_exponent_bits | 属性    | int64 | NA                                             | NA | FP8 反量化时使用的指数位数，若未启用 FP8 传 -1                   |
| fp8_exponent_bias | 属性    | int64 | NA                                             | NA | FP8 反量化时的 bias，若未启用 FP8 传 -1                    |
| out | 输出    | FP32/FP16/BF16/uint8 | bag: [B, total_D]；nobag: [len(indices), max_D] | NA | 查表结果；仅当所有表的权重类型为 INT8 时才允许 uint8 输出             |

# Embedding 维度约束

## FP8 量化权重约束
- **维度对齐**：embedding 维度（D）必须是 **4 的倍数**
- **维度上限**：D <= **4096**

## Nobag 模式特殊约束
- **维度一致性**：nobag 模式下，所有表的 embedding 维度必须相同，即 `D_offsets[i+1] - D_offsets[i]` 对所有表 i 都相等
- **维度值**：所有表的 D 必须等于 `max_D`（即 `D_offsets[-1] - D_offsets[0]`）

# 编译与部署
参考 [RecSDK/cust_op/README.md](../../../../README.md) “单算子使用说明”章节的编译、适配层部署流程。

更多 PyTorch 调用示例见 `framework/torch_plugin/torch_library/int_nbit_split_embedding_codegen_lookup_function/README.md`。
