# 使用 PyTorch 调用算子工程的方式

该样例脚本基于 PyTorch 2.1.0, Python 3.10.0 运行

## 环境依赖

在[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/cfe24a13b14e465ebbcf816ad6f73c9e)下载 mxrec-tf2 镜像，并创建容器。

## 运行样例算子

### 1. 编译算子工程

运行此样例前，请参考[部署算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0072.html)完成算子部署。

- 例如：

    ```bash
    ./custom_opp_<target os>_<target architecture>.run --install-path=<path>
    ```

### 2. 编译 PyTorch 自定义算子库

```bash
cd torch_library/common
bash build_ops.sh
```

- 说明：torch_library 目录下如果有多个算子，会将所有算子适配层编译成一个 .so 文件，并同时在 Python 默认的 site-packages 路径下存放编译好的 libfbgemm_npu_api.so 方便使用。如果只想编译单个算子的，只需进入具体算子适配层目录进行编译。例如：

```bash
cd torch_library/hstu
bash build_ops.sh
```

注：单算子编译后算子加载时需要引用 build 目录下的绝对路径

### 3. PyTorch 框架调用样例运行

- 样例执行

    样例执行过程中会自动生成测试数据，然后运行 PyTorch 样例，最后打印运行结果。详见 hstu_dense_forward_demo.py

    ```bash
    cd ../../
    python3 test_hstu_dense_forward.py
    ```

## 更新说明

| 时间        | 更新事项      |
|------------|--------------|
| 2024/12/23 | 新增本readme |
