<h1 align="center">Rec SDK</h1>

<div align="center">

 [![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/cn/developer/software/mindsdk)
 [![License](https://badgen.net/badge/License/Apache-2.0/blue)](LICENSE)
 [![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/RecSDK)
 [![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/RecSDK)

</div>

## ✨ What's New

<span style="font-size:14px;">

🔹 **[2026.07.31]**: [Rec SDK 26.1.0 Release](https://gitcode.com/Ascend/RecSDK/releases/v26.1.0)<br>
🔹 **[2026.04.25]**: [Rec SDK 26.0.0 Release](https://gitcode.com/Ascend/RecSDK/releases/v26.0.0)<br>

</span>

## ℹ️ Introduction

Rec SDK offers an SDK for search, recommendation, and advertising services in the Internet market. It offers a framework for these services based on the Ascend platform to meet related model training requirements, thus supporting large-scale search-recommendation-advertising scenarios and facilitating efficient training of models for such scenarios.

Rec SDK provides the following features:

1. Basic model training functions: Rec SDK supports single-node, single-card training and multi-node, multi-card distributed training.
2. Recommendation-specific functions: Based on the sparse table solution, Rec SDK provides essential functions such as feature saving and loading, feature admission, and feature eviction.
3. Large-scale sparse table functions: Rec SDK supports multi-level storage across accelerator card memory, host memory, and host drive. It also supports multi-node storage and dynamic scaling, with capacities exceeding 10 TB.

<img src="./docs/zh/figures/rec_full_stack/全栈架构图.png" width="1200"/>

## ⚙️ Features

| Component | Feature Summary | Documentation |
| --- | --- | --- |
| tf_rec_v1 | Supports single-/multi-node and single-/multi-card distributed training; provides recommendation-specific features such as feature saving and loading, feature admission and eviction, dynamic capacity expansion, dynamic shape, automatic graph rewriting, Hot_Embedding, customized WarmStart, incremental model saving/loading, single-table multi-query, and PCIe through; supports multi-level storage across accelerator memory, host memory, and host disk (capacity > 10 TB); provides performance and accuracy detection tools. | [Details](./docs/en/tensorflow/tf_rec_v1/introduction.md) |
| tf_rec_v2 | Supports single-/multi-node and single-/multi-card distributed training; provides recommendation-specific features such as sparse table creation, query, saving and loading, and feature admission and eviction; supports large-scale sparse table storage. | [Details](./docs/en/tensorflow/tf_rec_v2/introduction.md) |
| torch_rec_v1 | Supports single-/multi-node and single-/multi-card distributed training; provides recommendation-specific features such as hash mapping, EBC table lookup, Row-wise table sharding, pipelined lookup, and fused lookup operators; supports Row-wise distributed sparse table sharding. | [Details](./docs/en/torch/torch_rec_v1/introduction.md) |
| torch_rec_v2 | Supports single-/multi-node and single-/multi-card distributed training; provides recommendation-specific features such as hash mapping, Row-wise table sharding, dynamic sparse table expansion and eviction, and dynamic sparse table operators; implements dynamic sparse table operators based on the HKV high-performance key-value storage acceleration library. | [Details](./docs/en/torch/torch_rec_v2/introduction.md) |

## 🚀 Quick Start

| Component | Base Framework | Adaptation Status | Framework Type | Description | Documentation |
| --- | --- | --- | --- | --- | --- |
| tf_rec_v1 | TensorFlow | Non-fully-offloaded | Sparse recommendation framework | Non-fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices | [Quick Start](./docs/en/tensorflow/tf_rec_v1/quick_start.md) |
| tf_rec_v2 | TensorFlow | Fully-offloaded | Sparse recommendation framework | Fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices (PoC) | [Quick Start](./docs/en/tensorflow/tf_rec_v2/quick_start.md) |
| torch_rec_v1 | PyTorch + TorchRec | Non-fully-offloaded | Sparse recommendation framework | Non-fully-offloaded sparse recommendation framework based on open-source PyTorch and TorchRec, adapted for NPU devices | [Quick Start](./docs/en/torch/torch_rec_v1/quick_start.md) |
| torch_rec_v2 | PyTorch + TorchRec | Fully-offloaded | Sparse recommendation framework | Fully-offloaded sparse recommendation framework based on open-source PyTorch and TorchRec, adapted for NPU devices (PoC) | [Quick Start](./docs/en/torch/torch_rec_v2/quick_start.md) |

Key terminology:

- **Non-fully-offloaded**: A hybrid mode in which some computing tasks are executed on the NPU and others on the CPU.
- **Fully-offloaded**: A mode where all computing tasks are offloaded to the NPU for execution.
- **PoC**: Proof of Concept, indicating that the component is still in the experimental verification phase and features may be incomplete or unstable.

## 📦 Installation Guide

The following product models are supported by Rec SDK:

- Atlas 200T A2 Box16
- Atlas 800T A2 training server
- Atlas 900 A3 SuperPoD
- Ascend 950 series products

| Component | Installation Guide |
| --- | --- |
| tf_rec_v1 | [Installation Guide](./docs/en/tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md) |
| tf_rec_v2 | [Installation Guide](./docs/en/tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md) |
| torch_rec_v1 | [Installation Guide](./docs/en/torch/torch_rec_v1/recsdk_torch_installation_guide.md) |
| torch_rec_v2 | [Installation Guide](./docs/en/torch/torch_rec_v2/recsdk_torch_installation_guide.md) |

## 📘 Usage Guide

Rec SDK provides comprehensive usage and development documentation to help you understand the architecture, tuning, and usage of each component. For details, refer to the Ascend community [recommendation development documentation](https://www.hiascend.com/cn/developer/recommendation?tab=tab1).

**Mode adaptation examples**:

| Model | Framework | Component | Code link |
| --- | --- | --- | --- |
| DIN | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/din) |
| DLRM(DCNv2) | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_examples/dlrm) |
| GR | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/gr/gr_meta) |
| GR | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/gr_nv) |
| MMOE, ETA | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/blob/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/model_zoo/README.md) |
| GR | PyTorch | torch_rec_v2 | [Code link](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/gr) |

## 🗺️ Roadmap

[Roadmap (2026Q3)](https://gitcode.com/Ascend/RecSDK/issues/1280)

[Roadmap (2026Q2)](https://gitcode.com/Ascend/RecSDK/issues/1266)

[Roadmap (2026Q1)](https://gitcode.com/Ascend/RecSDK/issues/1075)

## 🔀 Version Maintenance Strategy

The maintenance stages of Rec SDK version branches are as follows:

| **Status** | **Duration** | **Description** |
| --- | --- | --- |
| Plan | 1–3 months | Planned features |
| Development | 6–12 months | Develop new features and fix issues, and release new versions regularly. Different strategies apply to different PyTorch versions: regular branches have a 6-month development cycle, and long-term support branches have a 12-month development cycle. |
| Maintenance | 1 year / 3.5 years | Regular branches are maintained for 1 year, and long-term support branches for 3.5 years. Major bugs are fixed, no new features are merged, and patch versions are released based on bug impact. |
| End of Life (EOL) | N/A | The branch no longer accepts any changes |

### torch_rec_v1 Version Maintenance

The maintenance status of each torch_rec_v1 version is as follows:

| **torch_rec_v1 version** | **Maintenance policy** | **Current status** | **Release date** | **Next status** | **EOL date** |
| --- | --- | --- | --- | --- | --- |
| 1.1.0 | Regular branch | Maintenance | July 23, 2025 | Expected to enter unmaintained status after September 30, 2026 | September 30, 2026 |
| 1.2.0 | Regular branch | Maintenance | January 4, 2026 | Under maintenance | |

## 🛠️ How to Contribute

We welcome your contributions. For the contribution process and specifications, see the [Contribution Guidelines](./contributing.md).
Before contributing, sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1).

1. If you encounter a bug, submit an [issue](https://gitcode.com/Ascend/RecSDK/issues).
2. If you plan to contribute bug fixes, submit a pull request (PR). See [Contribution Requirements](./contributing.md#pullrequest).
3. If you plan to contribute new features or functionality, create an issue to discuss it with us first. Describe the background or purpose of the requirement, the design, and its impact on existing APIs. Submitting a PR without prior discussion may lead to rejection, as the evolution direction of the project might differ from your ideas.

## ⚖️ Related Information

🔹 [Release notes](https://gitcode.com/Ascend/RecSDK/releases)<br>
🔹 [License](LICENSE)<br>
🔹 [Document license](./docs/LICENSE)<br>
🔹 [Disclaimer](docs/zh/disclaimer.md)<br>
🔹 Component-related information

| Component | FAQ | Security hardening |
| --- | --- | --- |
| tf_rec_v1 | [FAQ](./docs/en/tensorflow/tf_rec_v1/faq.md) | [Security hardening](./docs/en/tensorflow/tf_rec_v1/security_hardening.md) |
| tf_rec_v2 | [FAQ](./docs/en/tensorflow/tf_rec_v2/faq.md) | [Security hardening](./docs/en/tensorflow/tf_rec_v2/security_hardening.md) |
| torch_rec_v1 | / | [Security hardening](./docs/en/torch/torch_rec_v1/security_hardening.md) |
| torch_rec_v2 | / | [Security hardening](./docs/en/torch/torch_rec_v2/security_hardening.md) |

## 🤝 Suggestions and Communication

You are welcome to raise questions and join discussions through the following channels.

| Resource | Description |
| --- | --- |
| [Create an issue](https://gitcode.com/Ascend/RecSDK/issues/new) | Submit a bug, requirement, or suggestion |
| [Community tasks](https://gitcode.com/Ascend/RecSDK/issues/1096) | View and claim community tasks |
| [Meeting calendar](https://meeting.ascend.osinfra.cn/?sig=sig-RecSDK) | Regular community meetings and events |

## 🙏 Acknowledgments

Rec SDK is jointly contributed by the following Huawei departments:

- Ascend Computing Application Enablement Development Dept
- Software Platform Dept, Computing Product Line
- UnifiedBus Computing Cluster Development Dept
- Technology Development Dept, Computing Product Line
- Poisson Lab

Thank you to everyone in the community for your PRs. We warmly welcome your contributions to Rec SDK!
