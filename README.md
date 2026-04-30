# Rec SDK

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/RecSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/RecSDK)

</div>

## 最新消息

* [20260224] 资料结构整改，更新Roadmap（2026Q1）

### Roadmap

[Roadmap（2026Q1）](https://gitcode.com/Ascend/RecSDK/issues/1075)

## 简介

Rec SDK作为面向互联网市场搜索推荐广告的应用使能SDK产品，对于搜索推荐广告模型训练的应用场景需求，提供基于昇腾平台的搜索推荐广告框架，支撑大规模搜推广场景，助力完成搜推广模型的高效训练。

Rec SDK的功能涉及：

1. 模型训练基础功能。支持单机单卡训练、多机多卡分布式训练。
2. 推荐场景特有功能。基于Rec SDK的稀疏表方案，Rec SDK提供必备功能，如特征保存和加载、特征准入、特征淘汰等。
3. 大规模稀疏表特有功能。支持加速卡内存、主机内存、主机磁盘多级存储、支持多机存储、支持动态扩容。规模可超10TB。

## 目录结构

```text
RecSDK/                                          # 项目根目录
    |-- build/                                   # 构建脚本、生成的wheel包等
    |
    |-- cust_op/
    |    |-- ascendc_op                          # Ascend C编写，编译后在AI Core执行的算子和其编译脚本
    |    |-- framework                           # 算子适配层
    |    |-- hkv                                 # hkv子模块代码，项目代码来自https://gitcode.com/Ascend/HierarchicalKV-ascend.git
    |    |-- test                                # 算子测试用例
    |    |-- tf_cpu_op                           # CPU算子
    |
    |-- docs/                                    # 项目资料文档、镜像构建脚本、公网和邮箱地址及通信矩阵文档
    |
    |-- training
         |-- common                              # 公共组件
         |-- tf_rec_v1                           # 基于TensorFlow，适配NPU设备的非全下沉稀疏推荐框架
         |-- tf_rec_v2                           # 基于TensorFlow，适配NPU设备的全下沉稀疏推荐框架（POC状态）
         |-- torch_rec_v1                        # 基于PyTorch、TorchRec开源软件，适配NPU设备的非全下沉稀疏推荐框架
         |-- torch_rec_v2                        # 基于PyTorch、TorchRec开源软件，适配NPU设备的全下沉稀疏推荐框架（POC状态）
```

### 各组件说明

| 组件名称 | 基础框架 | 适配状态 | 框架类型 | 功能描述 |
|---------|---------|---------|---------|---------|
| tf_rec_v1 | TensorFlow | 非全下沉 | 稀疏推荐框架 | 基于TensorFlow，适配NPU设备的非全下沉稀疏推荐框架 |
| tf_rec_v2 | TensorFlow | 全下沉 | 稀疏推荐框架 | 基于TensorFlow，适配NPU设备的全下沉稀疏推荐框架（POC状态） |
| torch_rec_v1 | PyTorch + TorchRec | 非全下沉 | 稀疏推荐框架 | 基于PyTorch、TorchRec开源软件，适配NPU设备的非全下沉稀疏推荐框架 |
| torch_rec_v2 | PyTorch + TorchRec | 全下沉 | 稀疏推荐框架 | 基于PyTorch、TorchRec开源软件，适配NPU设备的全下沉稀疏推荐框架（POC状态） |

### 关键术语说明

- **非全下沉**：指部分计算任务在NPU上执行，部分在CPU上执行的混合模式
- **全下沉**：指所有计算任务都下沉到NPU上执行，以获得更好的性能
- **POC状态**：Proof of Concept（概念验证）状态，表示该组件仍处于试验验证阶段，功能可能不完整或不稳定

## 版本说明

通常，RecSDK一年会有4个正式release版本。

具体版本更新内容，参见：

* [releases](https://gitcode.com/Ascend/RecSDK/releases)

## 环境部署

支持的产品型号如下：

* Atlas 200T A2 Box16
* Atlas 800T A2 训练服务器
* Atlas 900 A3 SuperPoD 超节点

具体组件部署方式，参见：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md)
* [torch_rec_v1](./docs/zh/torch/torch_rec_v1/recsdk_torch_installation_guide.md)
* [torch_rec_v2](./docs/zh/torch/torch_rec_v2/recsdk_torch_installation_guide.md)

## 源码编译安装

参见具体组件：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md#源码编译安装)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md#源码编译安装)
* [torch_rec_v1](./docs/zh/torch/torch_rec_v1/recsdk_torch_installation_guide.md#源码编译安装)
* [torch_rec_v2](./docs/zh/torch/torch_rec_v2/recsdk_torch_installation_guide.md#源码编译安装)

## 快速入门

参见具体组件：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/quick_start.md)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/quick_start.md)
* [torch_rec_v1](./docs/zh/torch/torch_rec_v1/quick_start.md)
* [torch_rec_v2](./docs/zh/torch/torch_rec_v2/quick_start.md)

## 功能介绍&特性介绍

参见具体组件：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/introduction.md)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/introduction.md)
* [torch_rec_v1](./docs/zh/torch/torch_rec_v1/introduction.md)
* [torch_rec_v2](./docs/zh/torch/torch_rec_v2/introduction.md)

## API参考

参见具体组件：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/api)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/api)
* [torch_rec_v1](./docs/zh/torch/torch_rec_v1/api)
* [torch_rec_v2](./docs/zh/torch/torch_rec_v2/api)

## 模型适配样例

| 模型名称 | 适配框架 | 组件名称 | 说明 |
|---------|---------|---------|------|
| DIN | PyTorch | torch_rec_v1 | [代码链接](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/din) |
| DLRM(DCNv2) | PyTorch | torch_rec_v1 | [代码链接](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/dlrm) |
| GR | PyTorch | torch_rec_v1 | Facebook GR模型，[代码链接](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/gr/gr_meta) |
| GR | PyTorch | torch_rec_v1 | NVIDIA recsys-GR模型，[代码链接](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/gr_nv) |
| mmoe、eta | PyTorch | torch_rec_v1 | [代码链接](https://gitcode.com/Ascend/RecSDK/blob/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/model_zoo/README.md) |
| GR | PyTorch | torch_rec_v2 | NVIDIA recsys-GR模型，[代码链接](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/gr) |

## 分支维护策略

| 分支名 | 描述 | 维护状态 |
|---------|---------|---------|
| develop | 主开发分支 | 长期维护 |
| develop_examples_and_tools | demo，模型样例，模型相关工具 | 活跃维护 |
| develop_torch_benchmark | benchmark模型适配 | 活跃维护 |

## FAQ

参见具体组件：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/faq.md)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/faq.md)

## 贡献指导

贡献代码前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)。

1. 如果您遇到bug，请[提交issue](https://gitcode.com/Ascend/RecSDK/issues)。
2. 如果您计划贡献bug-fixes，请提交Pull Requests，参见[具体要求](./contributing.md#pullrequest)。
3. 如果您计划贡献新特性、功能，请先创建issue与我们讨论。写明需求背景/目的，如何设计，对现有API等的影响。未经讨论提交PR可能会导致请求被拒绝，因为项目演进方向可能与您的想法存在偏差。

## 联系我们

更详细的交流、贡献方式，请参考[贡献指南](./contributing.md)。

## 安全声明

用户应根据自身业务，重新审视整个系统的网络安全加固措施，必要时可参考业界优秀加固方案和安全专家的建议。

具体安全加固措施，参见具体组件：

* [tf_rec_v1](./docs/zh/tensorflow/tf_rec_v1/security_hardening.md)
* [tf_rec_v2](./docs/zh/tensorflow/tf_rec_v2/security_hardening.md)
* [torch_rec_v1](./docs/zh/torch/torch_rec_v1/security_hardening.md)
* [torch_rec_v2](./docs/zh/torch/torch_rec_v2/security_hardening.md)

## 免责声明

本代码仓库中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布之前，这些分支不应被用于任何生产环境或依赖关键业务的项目中。请务必仅使用我们的正式发行版本，以确保代码的稳定性和安全性。
使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。
正式版本请参考Rec SDK正式[release](https://gitcode.com/Ascend/RecSDK/releases)版本。

## License

Apache License Version 2.0，详见[LICENSE文件](./LICENSE)。
Rec SDK docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE文件](./docs/LICENSE)。

## 致谢

Rec SDK由华为公司的下列部门联合贡献：

* 昇腾计算应用使能开发部
* 计算软件平台部
* 灵衢算力集群开发部
* 计算技术开发部
* 泊松实验室

感谢来自社区的每一个PR，欢迎贡献Rec SDK！
