# RecSDK-Torch 自定义算子说明

在推荐训练中，存在部分算子无NPU实现，或已有NPU实现但性能较差，不能满足推荐训练需求。RecSDK提供了自定义算子用于支持或加速推荐模型NPU训练。其中部分自定义算子已绑定到开源API(将开源API的backend实现转发到NPU)
，导入RecSDK软件包后，可直接通过开源API调用到自定义算子。其余算子则需通过PTA层注册的mxrec模块进行调用。详情请参考各算子目录下的README文件。

说明：本说明文档只针对Torch框架下适配的推荐算子

## 算子文件结构

```shell
cust_op
├── ascendc_op
│   ├── ai_core_op   # AscendC 算子功能实现
│   ├── build        # 算子编译
│   ├── config       # 芯片型号归一化配置（transform.json）
│   └── scripts      # 算子构建辅助脚本
├── framework
│   ├── torch_plugin # Torch 算子适配层实现
│   └── tf_plugin    # TensorFlow 算子适配层实现
├── rec_cust_ops     # rec_cust_ops Python 包（import 入口，加载适配层 .so 并注册 OPP 路径）
├── tf_cpu_op        # TF CPU 算子
├── scripts          # whl 打包辅助脚本（custom_opp 提取等）
├── test             # 算子测试用例
├── build_whl.sh     # 统一构建入口（算子编译 + 适配层 + whl 打包）
├── setup.py         # pip 构建配置
├── CMakeLists.txt   # CMake 构建入口
└── RecOps.cmake     # AscendC 算子编译规则（新增算子只需改此文件）
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## 版本配套说明

当前支持两种软件版本配套：PyTorch 2.6.0和PyTorch2.7.1。调用算子前需完成配套软件的安装和所需算子的安装。详细配套关系如下：

| 配套版本  | PyTorch | TorchNPU | torchrec  | fbgemm_gpu | hybrid_torchrec |
|-------|---------|-----------|-----------|------------|-----------------|
| 配套版本1 | 2.7.1+cpu   | 2.7.1    | 1.2.0+npu | 1.2.0+cpu | 1.2.0         |
| 配套版本2 | 2.10.0+cpu  | 2.10.0   | 1.5.0+npu | 1.5.0+cpu | 1.5.0         |

## 编译自定义算子

在26.1.0及之前版本，torch_rec_v1框架中采用编译算子 + 算子适配层（libfbgemm_npu_api.so）的方式使用自定义算子。

在26.2.0及之后版本，torch_rec_v1框架中采用导入rec_cust_ops + [fbgemm_ascend](https://gitcode.com/Ascend/fbgemm-ascend)的方式使用自定义算子。

### 26.2.0及之后版本算子编译<a id="build_recsdk_cust_ops"></a>

26.2.0及之后版本的算子包包含rec_cust_ops和fbgemm_ascend两个包。fbgemm_ascend可直接从[fbgemm_ascend Release](https://gitcode.com/Ascend/fbgemm-ascend/releases)获取（fbgemm_ascend版本和PyTorch配套关系参考fbgemm_gpu版本即可），再通过如下指令进行安装。

```bash
pip3 uninstall -y fbgemm_ascend
pip3 install fbgemm_ascend-*-cp311-cp311-linux_*.whl
```

本章节主要介绍rec_cust_ops的编译和安装。通过 `build_whl.sh` 脚本可一键完成rec_cust_ops包的算子编译、torch_plugin适配层（.so）编译和 whl打包，产物统一输出到`dist/`目录。打包完成后，安装`dist/`目录下的whl包，即可通过`import rec_cust_ops`的方式导入自定义算子。torch_rec_v1框架包（hybrid_torchrec/torchrec_embcache）已默认导入fbgemm_ascend、rec_cust_ops模块。

rec_cust_ops包中包含的算子列表可参考 [RecOps.cmake](RecOps.cmake)文件中RECSDK_CUSTOM_OPS_A2/RECSDK_CUSTOM_OPS_A3/RECSDK_CUSTOM_OPS_A5变量定义。

#### 前置条件

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 拉取三方模块（部分算子依赖 CATLASS）
git submodule update --init --recursive
```

#### 基础使用

脚本无需任何必选参数，全部环境变量均有默认值，直接执行即可按默认配置（全芯片变体、算子间并行编译）构建出统一whl包。

```bash
cd RecSDK/cust_op
bash build_whl.sh
```

编译完成后，进入dist目录，安装rec_cust_ops*.whl包：

```bash
pip3 uninstall -y rec_cust_ops
pip3 install rec_cust_ops*.whl
```

#### 详细入参（环境变量）

| 环境变量 | 说明 | 默认值 | 示例 |
|---|---|---|---|
| `RECSDK_BUILD_VERS` | AscendC算子编译的芯片变体范围，逗号分隔。受所在环境中CANN平台信息限制，只能编译CANN支持的芯片。注意：torch_plugin 适配层 .so始终编译全部变体（A5/A3/A2），不受此变量影响 | `A2,A3,A5` | `RECSDK_BUILD_VERS=A2,A3 bash build_whl.sh` |
| `SERIAL_BUILD` | 算子编译串/并行开关。取 `ON/1/true/yes` 为串行（旧行为，最保守），取 `OFF/0/false/no` 为并行（不同算子之间并行编译，同名算子跨变体仍串行）；其他值告警并回退为OFF | 未设置时为OFF，即开启并行编译 | `SERIAL_BUILD=ON bash build_whl.sh` |
| `RECSDK_ASCEND_SERIAL_BUILD` | 串/并行开关的内部兼容变量，优先级低于 `SERIAL_BUILD` | 未设置时为OFF | 一般无需直接使用 |

优先级：`SERIAL_BUILD` > `RECSDK_ASCEND_SERIAL_BUILD` > 默认 OFF。

#### 典型场景

```bash
# 默认全量构建（A2/A3/A5，并行编译）
bash build_whl.sh

# 在 A2 机器上构建全量包
RECSDK_BUILD_VERS=A2,A3 bash build_whl.sh

# 小内存机器退回全串行编译，避免 OOM
SERIAL_BUILD=ON bash build_whl.sh
```

### 26.1.0及之前版本算子编译

下载[RecSDK](https://gitcode.com/Ascend/RecSDK)源码，按如下指令进行算子相关包的编译和安装：

```bash
# 编译算子前，若未配置CANN环境变量则使用如下指令进行配置
source /usr/local/Ascend/cann/set_env.sh

# 编译并安装算子包（Ascend-recsdk-npu-ops-\*-linux-\*.tar.gz）。
cd RecSDK/cust_op/ascendc_op/build
git submodule update --init --recursive  # 拉取三方模块
bash build_ai_core_op.sh A2

# 安装算子适配层（libfbgemm_npu_api.so）
cd ../../framework/torch_plugin/torch_library/common/
bash build_ops.sh
```

算子编译安装脚本 `build_ai_core_op.sh` 的参数需根据实际芯片架构配置，默认以 `A2` 为例。详细参数和更多说明请参考 [算子编译README](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/build/README.md)。

## 单算子使用说明

### 算子编译

进入指定算子的功能实现目录(ascendc_op/ai_core_op/目录下)，执行指令对算子进行编译和部署，默认编译安装Atlas A2训练系列产品AI Core类型。

```shell
git submodule update --init --recursive #部分算子存在三方库依赖，需要初始化submodule
bash run.sh
```

若指定 AI Core 类型编译：

```shell
bash run.sh --ai-core ai_core-<soc_version>
```

> AI处理器的型号 soc_version 请通过如下方式获取:
>
> - 在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，获取`Chip Name`信息。实际配置值为AscendChip Name，例如`Chip Name`取值为`xxxyy`，实际配置值为`Ascendxxxyy`。
>
> 基于同系列的AI处理器型号创建的算子工程，其基础功能（基于该工程进行算子开发、编译和部署）通用。

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 算子适配层编译

#### 单算子编译

进入算子适配层(framework/torch_plugin/torch_library/目录下)，并进入到指定的算子目录。执行算子适配层编译。

```shell
bash build_ops.sh
```

执行完上述命令后将在当前build目录生成xxx.so文件，调用算子时需先执行以下命令加载so文件：

```python
import torch
torch.ops.load_library("path/to/build/xxx.so")  #.so文件的绝对路径
```

#### 多算子编译

进入算子适配层目录`RecSDK/cust_op/framework/torch_plugin/torch_library/common`下，执行如下命令编译。

```shell
bash build_ops.sh
```

编译完成后，会在common/build目录下生成`libfbgemm_npu_api.so`，并同时在python默认的site-packages路径下存放编译好的`libfbgemm_npu_api.so`。<br>
该so包含`RecSDK/cust_op/framework/torch_plugin/torch_library/`目录下所有算子的适配层。

加载so：

```python
import sysconfig
import torch
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
```

## 算子测试用例

完成算子及其算子适配层准备后，可通过算子用例验证。进入指定算子的测试用例目录(cust_op/test/目录下/torch)，执行如下命令运行测试用例：

```bash
python3 -m pytest -x your_script.py
```

## 算子介绍

各算子实现的详细介绍见具体算子目录中readme说明。

注意：自定义算子为高性能计算，用户调用自定义算子时需自行确保输入的参数满足算子约束条件、参数类型、参数shape等要求，否则可能出现数组越界，显存不够等问题导致算子执行失败。

# FAQs

## `Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Python3_LIBRARIES)`

若编译时报错：`Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Python3_LIBRARIES)`

需检查python3的软连接是否正确创建。未创建时，可参考如下命令创建：

```bash
ln -s /usr/local/python3.11.0/bin/python3 /usr/bin/python3
```
