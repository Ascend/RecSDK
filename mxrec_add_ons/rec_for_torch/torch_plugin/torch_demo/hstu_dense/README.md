## 使用pytorch的调用的方式调用算子工程
该样例脚本基于pytorch 2.1.0,python3.10.0运行

### 环境依赖
在[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/cfe24a13b14e465ebbcf816ad6f73c9e)下载mxrec-tf2镜像，并创建容器；

## 运行样例算子
### 1.编译算子工程
运行此样例前，请参考[部署算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0072.html)完成算子部署。
- 例如：
    ```bash
    ./custom_opp_<target os>_<target architecture>.run --install-path=<path>
    ```
### 2.编译pytorch自定义算子库
```bash
    cd torch_library/common
    bash build_ops.sh
```
- 说明:torch_library目录下如果有多个算子,会将所有算子适配层编译成一个.so文件,并同时在python默认的site-packages路径下存放编译好的libfbgemm_npu_api.so方便使用。如果只想编译单个算子的,只需进入具体算子适配层目录进行编译。例如：
```bash
    cd torch_library/hstu
    bash build_ops.sh
```
注:单算子编译后算子加载时需要引用build目录下的绝对路径


### 2.pytorch调用的方式调用样例运行
  - 样例执行

    样例执行过程中会自动生成测试数据，然后运行pytorch样例，最后打印运行结果。详见hstu_dense_forward_demo.py
    ```bash
    cd ../../
    python3 test_hstu_dense_forward_demo.py
    ```
## 更新说明
| 时间         | 更新事项     |
|------------| ------------ |
| 2024/12/23 | 新增本readme |