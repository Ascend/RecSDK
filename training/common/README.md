#  training common 说明

本目录包含RecSDK公共的组件

## 目录结构概览

```
├── python  # python公共组件
    ├── communication # 通信相关信息
    ├── constants # 公共常量
    ├── log # 日志模块
    ├── perf_factory # 优化模块
    ├── util # 公共util
    ├── validator # 检核模块
    ├── tests # 测试用例
├── src # C++公共组件
    ├── core # 核心组件
        ├── common_func # 公共函数库
        ├── error # 报错模块
        ├── initializer # 初始化器模块
        ├── lccl # lccl通信模块
        ├── log # 日志模块
        ├── pcie_through # pcie_through模块
        ├── tests # 测试模块
    ├── pybind # pybind接口
    ├── tests # 测试用例
```

## 源码编译安装

编译环境依赖：
- Python3.7.5
- GCC 7.3.0
- CMake 3.20.6

开源依赖：
- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)
- [openmpi 4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz): 请参考软件文档在编译环境完成安装
- tensorflow 1.15/2.6.5：根据实际需求选择对应版本


将pybind11和securec的压缩包放在与Rec SDK代码同级的opensource目录下，并且将其分别更名为pybind11-2.10.3.zip、huaweicloud-sdk-c-obs-3.23.9.zip。

**编译方法**
方案1：进入common代码目录，构建单独的rec_sdk_common whl包：
- src/build.sh：用于构建src下面的so文件
- setup.py：执行脚本 `python3.7 setup.py bdist_wheel` 构建whl包

方案2：在最外层的RecSdk目录，和mx_rec whl包一起打包
- setup_tf1.py：执行脚本 `python3.7 setup_tf1.py bdist_wheel` 构建tf1版本whl包。构建成功后，whl包在build/mindxsdk-mxrec/tf1_whl子目录下。
- setup_tf2.py：执行脚本 `python3.7 setup_tf2.py bdist_wheel` 构建tf2版本whl包。构建成功后，whl包在build/mindxsdk-mxrec/tf2_whl子目录下。

## 测试用例
> 运行测试用例前需先设置CANN相关环境变量，参考前文`安装方式`章节。

### Python侧测试用例

运行Python测试用例所需依赖：

- pytest 7.1.1
- pytest-cov 4.1.0
- pytest-html

如需使用python测试用例，需要先安装上述依赖以及能够在tf1环境下进行源码编译并安装好rec_sdk_common，然后进入tests目录中。参考以下命令执行python侧测试用例：
```shell
bash run_python_dt.sh
```

### C++侧测试用例

运行C++侧测试用例所需依赖：

- [googletest 1.8.1](https://github.com/google/googletest/archive/refs/tags/release-1.8.1.zip)
- [emock 0.9.0](https://github.com/ez8-co/emock/archive/refs/tags/v0.9.0.zip)
- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)

将googletest、emock、pybind11和securec的压缩包放在与Rec SDK代码同级的opensource目录下，并且将其分别更名为googletest-release-1.8.1.zip、
emock-0.9.0.zip、pybind11-2.10.3.zip、 huaweicloud-sdk-c-obs-3.23.9.zip。

如需使用C++测试用例，需要按照上述描述准备需要的依赖，准备好之后，进入src目录中。参考以下命令执行C++测试用例：

使用如下命令：
```shell
bash test_ut.sh
```

注：
1. 部分c++用例使用了emock库进行打桩，需要在x86环境上运行；在aarm64环境运行失败时可忽略。
2. test_ut.sh脚本运行完成后会使用[lcov](https://github.com/linux-test-project/lcov/releases/download/v1.13/lcov-1.13.tar.gz)及相关工具（如perl-Digest-MD5）生成覆盖率统计信息，若指令未安装完全生成失败时可忽略。