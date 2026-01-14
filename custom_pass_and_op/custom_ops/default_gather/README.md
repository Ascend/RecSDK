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
-- default_gather
   |-- v220
      |-- op_host    # default_gather算子Host侧实现
      |-- op_kernel  # default_gather算子Kernel侧实现
      |-- default_gather.json    # 算子原型配置
      |-- run.sh     # default_gather算子A2安装脚本
   |-- v330
      |-- run.sh # default_gather算子A5安装脚本
   |-- README.md  # default_gather算子说明文档
```

# 功能

推荐场景下，针对带缺省值（通过<0的索引表达）的特征实现embedded查表功能, 当前只支持按0轴查询。


# 算子输入与输出

| 名称                | 输入/输出 | 数据类型                             | 
|---------------------|----------|-------------------------------------|
| id                 | 输入    | Tensor[int32/int64] |
| table              | 输入    | Tensor[float32/float32] | 
| attn_output        | 输出    | Tensor[float32/float32] | 


# 算子编译部署

基于A2/A5分别进入v220/v330目录下bash run.sh一键安装。
算子使用时，需要参考https://www.hiascend.com/昇腾官网上资料，配置ASCEND_CUSTOM_OPP_PATH环境变量，当前算子路径为<CANN路径>/opp/vendors/default_gather
参考配置：
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}:${ASCEND_HOME_PATH}/opp/vendors/default_gather:${ASCEND_HOME_PATH}/opp/vendors/default_gather/op_api/lib/