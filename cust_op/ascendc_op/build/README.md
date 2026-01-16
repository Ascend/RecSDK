### 依赖组件准备
部分算子依赖外部组件，编译前请将组件 [v3.9.1.tar.gz](https://github.com/nlohmann/json/archive/v3.9.1.tar.gz) 下载后放置于 RecSDK/cust_op/ascendc_op/build/scripts/onnx_plugin 目录

### 编译生成 Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz
```
cd RecSDK/cust_op/ascendc_op/build
bash build_ai_core_op.sh [ver]

e.g. 'bash build_ai_core_op.sh A2'
```

生成的tar包在 RecSDK/cust_op/ascendc_op/output 下

**安装方法**

```
tar zxvf Ascend-recsdk-npu-ops-*-linux-*.tar.gz
cd recsdk-npu-ops/recsdk_ops

# 安装所需算子
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
# ...
```