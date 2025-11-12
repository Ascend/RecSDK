# RecSDK-Torch 自定义算子说明
在推荐训练中，存在部分算子无NPU实现，或已有NPU实现但性能较差，不能满足推荐训练需求。RecSDK提供了自定义算子用于支持或加速推荐模型NPU训练。其中部分自定义算子已绑定到开源API(将开源API的backend实现转发到NPU)
,导入RecSDK软件包后，可直接通过开源API调用到自定义算子。其余算子则需通过PTA层注册的mxrec模块进行调用。详情请参考各算子目录下的README文件。

说明：本说明文档只针对Torch框架下适配的推荐算子

## 算子文件结构

```shell
├── ascendc_op
    ├──ai_core_op   # 算子功能实现
    ├──build        # 算子编译
├── framework
    ├──torch_plugin # 算子适配层实现
├── test            # 算子测试用例
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## 版本配套说明
当前支持两种软件版本配套：PyTorch 2.6.0和PyTorch2.7.1。调用算子前需完成配套软件的安装和所需算子的安装。详细配套关系如下：

| 配套版本  | PyTorch | torch-npu | torchrec  | fbgemm_gpu | hybrid_torchrec |
|-------|---------|-----------|-----------|------------|-----------------|
| 配套版本1 | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           |
| 配套版本2 | 2.7.1   | 2.7.1     | 1.2.0+npu | 1.2.0      | 1.2.0           |

**后续说明以PyTorch 2.6.0配套版本为例进行说明。**

## 单算子使用说明
### 算子编译

进入指定算子的功能实现目录(ascendc_op/ai_core_op/目录下)，执行指令对算子进行编译和部署，默认编译安装Atlas A2训练系列产品AI Core类型。

```shell
bash run.sh
```

若指定 AI Core 类型编译：

```shell
bash run.sh ai_core-<soc_version>
```

> AI处理器的型号<soc_version>请通过如下方式获取:
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

执行完在当前build目录生成 xxx.so文件,调用算子时执行以下命令进行加载。

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

> 若编译时报错：`Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Python3_LIBARIES)`
>
> 需注释`recsdk-npu-ops/torch_plugin/torch_library/common/CMakeLists.txt`中如下两行内容重新编译
>
> ```
> #find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
> #include_directories(${Python3_INCLUDE_DIRS})
> ```

加载so：
```python
import sysconfig
import torch
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
```

## 算子介绍
各算子实现的详细介绍见具体算子目录中readme说明。

须知：自定义算子为高性能计算，用户调用自定义算子时需自行确保输入的参数满足算子约束条件、参数类型、参数shape等要求，否则可能出现数组越界，显存不够等问题导致算子执行失败。