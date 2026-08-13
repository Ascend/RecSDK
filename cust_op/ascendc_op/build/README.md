# 统一编译脚本说明

build_ai_core_op.sh 脚本用于统一编译Ascend平台的RecSDK定制算子，支持A2、A2-TF、A3、A5、310P版本。

## 编译生成 Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz 并同时安装

```shell
cd RecSDK/cust_op/ascendc_op/build
bash build_ai_core_op.sh [ver] [rebuild_all] [error_mode]

e.g. 'bash build_ai_core_op.sh A2 false continue' 表示编译A2版本算子，跳过已编译成功的算子，并在编译过程中遇到错误时记录失败的算子但继续编译剩余算子
```

参数说明：

- ver：编译版本，支持A2、A2-TF、A3、A5、310P，必填项
- rebuild_all：是否重新编译已编译成功的算子，默认为true，设置为false后会跳过已编译成功的算子，减少编译时间，选填项
- error_mode：编译错误处理模式，默认为exit，设置为exit表示在编译过程中遇到错误时立即退出脚本，设置为continue表示在编译过程中遇到错误时记录失败的算子但继续编译剩余算子，选填项

编译后生成的tar包在 RecSDK/cust_op/ascendc_op/output 下，同时会自动安装到当前环境中。

编译耗时说明（耗时可能随CPU、内存等因素变化）：

- A2/A3：耗时约20~30min
- A2-TF/310P：耗时约5~10min
- A5：耗时约40-60min

## 依赖说明

### json库依赖说明

部分算子依赖外部组件，编译前请将组件 [v3.9.1.tar.gz](https://github.com/nlohmann/json/archive/v3.9.1.tar.gz) 下载后放置于 RecSDK/cust_op/ascendc_op/build/scripts/onnx_plugin 目录

依赖json库算子列表（仅310P版本时需要）：

- gather_for_rank1
- hstu_dense_forward_fuxi

### CATLASS依赖说明

部分算子依赖CATLASS第三方库，build_ai_core_op脚本在编译时会自动拉取第三方库，若拉取失败则会跳过这些算子的编译

依赖CATLASS算子列表：

- InLinearSilu
- HSTU_v2

## 取安装包并安装方法

也可以直接使用编译生成的安装包进行安装，安装方法如下：

```shell
tar zxvf Ascend-recsdk-npu-ops-*-linux-*.tar.gz
cd recsdk-npu-ops/recsdk_ops

# 安装所需算子
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
# ...
```
