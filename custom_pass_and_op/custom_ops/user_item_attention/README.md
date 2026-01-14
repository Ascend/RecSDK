**说明**

本算子仅支持NPU调用

# 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A2训练系列产品  | 是  |
| Atlas A3训练系列产品  | 是  |
| Atlas 推理系列产品    | 是  |

# default_gather算子文件结构

```shell
-- user_item_attention
   |-- v220
      |-- op_host    # user_item_attention算子Host侧实现
      |-- op_kernel  # user_item_attention算子Kernel侧实现
      |-- user_item_attention.json    # 算子原型配置
      |-- run.sh     # user_item_attention算子A2安装脚本
   |-- v330
      |-- run.sh # user_item_attention算子A5安装脚本
   |-- README.md  # user_item_attention算子说明文档
```

# 功能

推荐场景下，针对候选items对user历史序列的CrossAttention, 或针对(items+user)的SelfAttention拆分后的items对(items+user)的Attetion基于user bs为1，items seq_len为1的特性进行算法优化的定制Attention算子。


# 算子输入与输出

| 名称                | 输入/输出 | 数据类型                             | 说明               |
|--------------------|---------- |-------------------------------------|                    |
| query              | 输入    | Tensor[float16/bfloat16/float32] | 候选items矩阵                             |
| key_user           | 输入    | Tensor[float16/bfloat16/float32] | user历史序列矩阵的k投影                    |
| value_user         | 输入    | Tensor[float16/bfloat16/float32] | user历史序列矩阵的v投影                    |
| mask_len           | 输入    | Tensor[int32/int32/int32]        | 历史序列的有效长度                    |
| key_item           | 可选输入 | Tensor[float16/bfloat16/float32] | item特征的k投影                    |
| value_item         | 可选输入 | Tensor[float16/bfloat16/float32] | item特征的v投影                    |
| attn_out           | 输出    | Tensor[float16/bfloat16/float32] | attention输出                      |


# 算子编译部署

基于A2/A5分别进入v220/v330目录下bash run.sh一键安装。
算子使用时，需要参考https://www.hiascend.com/昇腾官网上资料，配置ASCEND_CUSTOM_OPP_PATH环境变量，当前算子路径为<CANN路径>/opp/vendors/user_item_flash_attention
参考配置：
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}:${ASCEND_HOME_PATH}/opp/vendors/user_item_flash_attention:${ASCEND_HOME_PATH}/opp/vendors/user_item_flash_attention/op_api/lib/