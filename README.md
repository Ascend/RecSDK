# mxRec

## 产品背景

随着人工智能技术的演进，电商、长短视频、社交等行业对搜索系统、推荐系统以及广告系统的效果诉求越发强烈。在如今互联网发达的时代，大量的用户数据、商品数据、视频资料，使信息剧烈爆炸，也使得搜索推荐广告系统的价值进一步凸显。搜索推荐广告系统的需求增长必然带来对算力的需求，如何部署更大算力并充分发挥算力成为系统管理人员重点关注的问题。

## 产品定义

mxRec作为面向互联网市场搜索推荐广告的应用使能SDK产品，对于搜索推荐广告模型训练的应用场景需求，提供基于昇腾平台的搜索推荐广告框架，支撑大规模搜推广场景，助力完成搜推广模型的高效训练。mxRec的功能涉及：

1. 模型训练基础功能。支持单机单卡训练、多机多卡分布式训练，支持基于TensorFlow开发模型。
2. 推荐场景特有功能。基于mxRec的稀疏表方案，mxRec提供必备功能，如特征保存和加载、特征准入、特征淘汰等。
3. 大规模稀疏表特有功能。支持加速卡内存、主机内存、主机磁盘多级存储、支持多机存储、支持动态扩容。规模可超10TB。

## 安装方式

安装前，请参考《CANN 软件安装指南CANN 软件安装指南》安装CANN开发套件软件包和TensorFlow适配昇腾插件。

CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。可在程序启动的Shell脚本中使用如下命令设置CANN的相关环境变量，也可通过命令行执行如下命令（以root用户默认安装路径“/usr/local/Ascend”为例）：
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
```

安装依赖，若未构建镜像，直接在物理机上进行开发，则须安装以下Python依赖
```shell
pip3 install numpy decorator sympy==1.4 cffi==1.12.3 pyyaml pathlib2 grpcio grpcio-tools protobuf==3.20.0 scipy requests mpi4py easydict scikit-learn==0.20.0 attrs
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

Wheel包默认安装在Python的“site-packages”路径，如通过“--target”参数指定目录，在安装完成后需要将mxRec路径加入“PYTHONPATH”环境变量。

```shell
export PYTHONPATH={mxrec_install_path}:{mxrec_install_path}/mxRec:$PYTHONPATH
```

如需使用动态扩容功能，进入已解压的mxRec软件包“mindxsdk-mxrec/cust_op/cust_op_by_addr”目录中。参考以下命令编译并安装动态扩容算子包。
```shell
bash run.sh
```

### 源码编译安装

编译环境依赖：
- Python3.7.5
- GCC 7.3.0
- CMake 3.20.6

开源依赖：
- pybind11 v2.10.3
- securec
- openmpi 4.1.1: 请参考软件文档在编译环境完成安装
- tensorflow 1.15/2.6.5：根据实际需求选择对应版本

pybind11的压缩包放在与MxRec代码同级的opensource/opensource目录下，如果没有opensource目录，则需要在MxRec同级的目录下手动创建opensource/opensource目录。然后将pybind11的压缩包放在opensource/opensource目录下。解压压缩包，并且将解压之后的压缩包改名为pybind11。

securec是华为开源的安全函数库。下载后：
1. 将platform下的eSDK_LogAPI_V2.1.10文件夹删除
2. 将platform下的huaweisecurec改名为securec
3. 在securec文件夹下，有src、lib和include三个文件夹，删除lib文件夹下的所有文件
4. 将platform文件夹放到MxRec代码目录下

为了构建多个版本的whl包，编译脚本在python虚拟环境完成对应tensorflow版本的安装。用户可以根据实际情况调整编译脚本，指定tensorflow的安装路径。编译方法：
- build/build.sh：执行脚本完成tf1和tf2版本whl包的构建和打包。执行脚本前，请参考build/build_tf1.sh、build/build_tf2.sh创建对应的虚拟环境，在虚拟环境中完成对应tensorflow版本的安装，并修改对应的激活命令。
- build/build_tf1_with_opensource.sh：执行脚本完成tf1版本whl包的构建，构建成功后，whl包在tf1_whl子目录下。执行脚本前，创建tf1虚拟环境，在虚拟环境中完成tensorflow 1.15.0版本的安装，并修改对应的激活命令。
- build/build_tf2_with_opensource.sh：执行脚本完成tf2版本whl包的构建，构建成功后，whl包在tf2_whl子目录下。执行脚本前，创建tf2虚拟环境，在虚拟环境中完成tensorflow 2.6.5版本的安装，并修改对应的激活命令。

如需使用动态扩容功能，进入“./cust_op/cust_op_by_addr”目录中。参考以下命令编译并安装动态扩容算子包。
```shell
bash run.sh
```

## 使用指导

mxRec所支持的使用环境、功能特性、API接口与使用样例请参考昇腾开源社区MindX SDK产品文档。

## 参考设计

mxrec框架基础镜像，基于TensorFlow 1.15.0、tensorflow2.6.5制作的基础镜像，安装mxrec后即可开始训练，以及样例使用介绍。

1. https://ascendhub.huawei.com/#/detail/mxrec-tf1

2. https://ascendhub.huawei.com/#/detail/mxrec-tf2
