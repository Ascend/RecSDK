# 使用pytorch的方式调用算子工程
该样例脚本基于python3.11,torch支持2.6.0版本。调用算子前需完成配套软件的安装和所需算子的安装。目前支持的torch版本配套关系如下：

| torch版本      | 配套关系                                                                                          |
|--------------|-----------------------------------------------------------------------------------------------|
| torch==2.6.0 | torch_npu==2.6.0<br/>fbgemm+gpu==1.1.0+cpu<br/>torchrec==1.1.0+npu<br/>hybrid_torchrec==1.1.0 |

## 运行样例算子
### 1.编译和安装自定义算子
- 编译Ascend-recsdk-npu-ops-*-linux-*.tar.gz软件包，编译方式：RecSDK/cust_op/ascendc_op/build/README.md
- 解压压缩包后，在recsdk-npu-ops/recsdk_ops/目录下安装执行所需的算子。

以下算子支持单算子直调：
```bash
    # 按需安装算子
    cd recsdk-npu-ops/recsdk_ops/
    bash mxrec_opp_asynchronous_complete_cumsum.run
    bash mxrec_opp_dense_to_jagged.run
    bash mxrec_opp_gather_for_rank1.run
    bash mxrec_opp_hstu_dense_backward.run
    bash mxrec_opp_hstu_dense_backward_fuxi.run
    bash mxrec_opp_hstu_dense_forward.run
    bash mxrec_opp_hstu_dense_forward_fuxi.run
    bash mxrec_opp_index_select_for_rank1_backward.run
    bash mxrec_opp_jagged_to_padded_dense.run
    bash mxrec_opp_relative_attn_bias_backward.run
    bash mxrec_opp_relative_attn_bias_pos.run
    bash mxrec_opp_relative_attn_bias_time.run
```

以下算子请通过torchrec接口调用：
```
  mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
  mxrec_opp_permute2d_sparse_data.run
  mxrec_opp_split_embedding_codegen_forward_unweighted.run
```

### 2.安装算子适配层

  - 进入到样例目录,执行如下命令。
    ```bash
    cd recsdk-npu-ops/torch_plugin/torch_library/2.6.0/common/
    bash build_ops.sh
    ```
  - 执行命令后会在common目录下生成libfbgemm_npu_api.so文件，并同时在python默认的site-packages路径下存放编译好的libfbgemm_npu_api.so方便使用。
    > 若编译时报错：`Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Python3_LIBARIES)`
    >
    > 需注释`recsdk-npu-ops/torch_plugin/torch_library/2.6.0/common/CMakeLists.txt`中5-7行内容重新编译
    >
    > ```
    > #find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    >
    > #include_directories(${Python3_INCLUDE_DIRS})
    > ```

### 3.样例执行

  - 样例执行时通过torch.ops.load_library的方式加载适配层文件。以下样例以python默认路径为例。
    ```bash
    import sysconfig
    import torch
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
    ```
  - 说明:common目录下的编译会将版本目录的所有算子适配层编译成一个.so文件,方便使用。如果只想编译单个算子的.so,只需进入具体算子适配层目录进行编译。例如：
    ```bash
    # 编译
    cd recsdk-npu-ops/torch_plugin/load_library/2.6.0/permute2d_sparse_data/
    bash build_ops.sh
    
    # 加载
    import sysconfig
    import torch
    torch.ops.load_library("path/to/build/libpermute2d_sparse_data.so") #.so文件的绝对路径
    ```

## 更新说明
| 时间         | 更新事项     |
|------------|----------|
| 2025/05/08 | 新增readme |