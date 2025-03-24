# Rec SDK

## 免责声明

本代码仓库中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布之前，这些分支不应被用于任何生产环境或依赖关键业务的项目中。请务必仅使用我们的正式发行版本，以确保代码的稳定性和安全性。
使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。
正式版本请参考Rec SDK正式release版本 https://gitee.com/ascend/mxrec/releases

## 产品背景

随着人工智能技术的演进，电商、长短视频、社交等行业对搜索系统、推荐系统以及广告系统的效果诉求越发强烈。在如今互联网发达的时代，大量的用户数据、商品数据、视频资料，使信息剧烈爆炸，也使得搜索推荐广告系统的价值进一步凸显。搜索推荐广告系统的需求增长必然带来对算力的需求，如何部署更大算力并充分发挥算力成为系统管理人员重点关注的问题。

## 产品定义

Rec SDK作为面向互联网市场搜索推荐广告的应用使能SDK产品，对于搜索推荐广告模型训练的应用场景需求，提供基于昇腾平台的搜索推荐广告框架，支撑大规模搜推广场景，助力完成搜推广模型的高效训练。Rec SDK的功能涉及：

1. 模型训练基础功能。支持单机单卡训练、多机多卡分布式训练，支持基于TensorFlow开发模型。
2. 推荐场景特有功能。基于Rec SDK的稀疏表方案，Rec SDK提供必备功能，如特征保存和加载、特征准入、特征淘汰等。
3. 大规模稀疏表特有功能。支持加速卡内存、主机内存、主机磁盘多级存储、支持多机存储、支持动态扩容。规模可超10TB。

## 版本配套

| 软件                | 版本                | 下载链接                                                                                                                  |
|-------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------|
| Rec SDK           | 6.0.RC3           | https://gitee.com/ascend/RecSDK/releases/tag/6.0.0                                                                 |
| CANN              | 8.0.RC3.alpha003  | https://www.hiascend.com/developer/download/community/result?module=tf+cann&tf=8.0.0.alpha003&cann=8.0.0.alpha003  |
| TensorFlowAdapter | 8.0.RC3.alpha003  | https://www.hiascend.com/developer/download/community/result?module=tf+cann&tf=8.0.0.alpha003&cann=8.0.0.alpha003  |
| Driver            | 23.0.0            | https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=All&driver=1.0.27.alpha       |
| Firmware          | 23.0.0            | https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=All&driver=1.0.27.alpha       |

## 支持的产品型号
- Atlas 200T A2 Box16
- Atlas 800T A2 训练服务器

## 安装方式

安装前，请参考[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)安装CANN开发套件软件包和TensorFlow适配昇腾插件。

CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。可在程序启动的Shell脚本中使用如下命令设置CANN的相关环境变量，也可通过命令行执行如下命令（以root用户默认安装路径“/usr/local/Ascend”为例）：
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
```

安装依赖，若未构建镜像，直接在物理机上进行开发，则须安装以下Python依赖
```shell
pip3 install numpy decorator sympy==1.4 cffi==1.12.3 pyyaml pathlib2 pandas grpcio grpcio-tools protobuf==3.20.0 scipy requests mpi4py easydict scikit-learn==0.20.0 attrs
```

horovod依赖安装前需配置“HOROVOD_WITH_MPI”、“HOROVOD_WITH_TENSORFLOW”，依赖安装命令参考如下。
```shell
HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip3.7 install horovod --no-cache-dir
```

### 二进制包安装

从昇腾开源社区直接获取编译打包后的产品包。解压后包含tf1和tf2两个版本的whl安装包，使用pip命令安装whl包（请根据实际需求，选取对应TensorFlow版本匹配的Wheel包）：
```shell
pip3 install mx_rec-{version}-py3-none-linux_{arch}.whl
```

Wheel包默认安装在Python的“site-packages”路径，如通过“--target”参数指定目录，在安装完成后需要将安装路径加入“PYTHONPATH”环境变量。

```shell
export PYTHONPATH={rec_install_path}:{rec_install_path}/mx_rec:$PYTHONPATH
```

如需使用动态扩容功能，进入已解压的Rec SDK软件包“mindxsdk-mxrec/cust_op/cust_op_by_addr”目录中。参考以下命令编译并安装动态扩容算子包。
```shell
bash run.sh
```

### 源码编译安装

编译环境依赖：
- Python3.7.5
- GCC 7.3.0
- CMake 3.20.6

开源依赖：
- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)
- [openmpi 4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz): 请参考软件文档在编译环境完成安装
- tensorflow 1.15/2.6.5：根据实际需求选择对应版本

将pybind11和securec的压缩包放在与Rec SDK代码同级的opensource目录下，并且将其分别更名为pybind11-2.10.3.zip、huaweicloud-sdk-c-obs-3.23.9.zip。如果没有opensource目录，则需要在Rec SDK同级的目录下手动创建opensource目录，然后将pybind11和securec的压缩包放在opensource目录下。

由于构建脚本需要适配内部构建工程，所以在脚本中存在适配代码，但是这些代码可能对于用户来说不需要，所以在编译之前需要做如下处理：

在build目录中存在build_tf1.sh和build_tf2.sh，其中分别存在如下代码：
```shell
# 配置tf1路径
source /opt/buildtools/tf1_env/bin/activate
tf1_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow_core
deactivate tf1_env
```
```shell
# 配置tf2路径
source /opt/buildtools/tf2_env/bin/activate
tf2_path=$(dirname "$(dirname "$(which python3.7)")")/lib/python3.7/site-packages/tensorflow
deactivate tf2_env
```

可以看到，上述代码中都有激活Python虚拟环境的步骤，因此用户有两种选择：

1. 根据需要在/opt/buildtools/目录下（没有此目录需要先创建）创建tf1_env和tf2_env两个Python虚拟环境，并在虚拟环境中安装对应版本的Tensorflow
2. 将source /opt/buildtools/tf1_env/bin/activate和deactivate tf1_env注释掉或者删除或者source /opt/buildtools/tf2_env/bin/activate和deactivate tf2_env注释掉或者删除


编译方法：

进入Rec SDK代码目录：
- setup.py：此脚本供内部使用，用于同时构建tf1和tf2的Rec SDK包，用户通常只需要其中一个，所以建议使用下面两个脚本构建。
- setup_tf1.py：执行脚本setup_tf1.py，比如：**python3.7 setup_tf1.py bdist_wheel**完成tf1版本whl包的构建，构建成功后，whl包在build/mindxsdk-mxrec/tf1_whl子目录下。
- setup_tf2.py：执行脚本setup_tf2.py，比如：**python3.7 setup_tf2.py bdist_wheel**完成tf2版本whl包的构建，构建成功后，whl包在build/mindxsdk-mxrec/tf2_whl子目录下。

如需使用动态扩容功能，进入“./cust_op/cust_op_by_addr”目录中。参考以下命令编译并安装动态扩容算子包。
```shell
bash run.sh
```

## 测试用例

### Python侧测试用例

运行Python测试用例所需依赖：

- pytest 7.1.1
- pytest-cov 4.1.0
- pytest-html

如需使用python测试用例，需要先安装上述依赖以及能够在tf1环境下进行源码编译，然后进入tests目录中。参考以下命令执行python侧测试用例：
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
emock-0.9.0.zip、pybind11-2.10.3.zip、 huaweicloud-sdk-c-obs-3.23.9.zip。如果没有opensource目录，则需要在Rec SDK同级的目录下手动创建opensource目录，
然后将前述几个压缩包放在opensource目录下。

如需使用C++测试用例，需要按照上述描述准备需要的依赖，准备好之后，进入src目录中。参考以下命令执行C++测试用例：

tf1环境下使用如下命令：
```shell
bash test_ut.sh tf1
```

tf2环境下使用如下命令：
```shell
bash test_ut.sh tf2
```

## 使用指导

Rec SDK所支持的使用环境、功能特性、API接口与使用样例请参考[Rec SDK用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc3/mxRec/mxrecug/mxrecug_0004.html)。

## 样例

Rec SDK框架基础镜像，基于TensorFlow 1.15.0、tensorflow2.6.5制作的基础镜像，安装Rec SDK后即可开始训练，以及样例使用介绍。

1. https://www.hiascend.com/developer/ascendhub/detail/mxrec-tf1

2. https://www.hiascend.com/developer/ascendhub/detail/mxrec-tf2
