# Rec SDK

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/RecSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/RecSDK)

</div>

## What's New

* [20260224] Documentation restructuring and updated Roadmap (2026Q1)

### Roadmap

[Roadmap (2026Q1)](https://gitcode.com/Ascend/RecSDK/issues/1075)

## Introduction

Rec SDK offers an SDK for search, recommendation, and advertising services in the Internet market. It offers a framework for these services based on the Ascend platform to meet related model training requirements, thus supporting large-scale search-recommendation-advertising scenarios and facilitating efficient training of models for such scenarios.

Rec SDK provides the following features:

1. Basic model training functions: Rec SDK supports single-node, single-card training and multi-node, multi-card distributed training.
2. Recommendation-specific functions: Based on the sparse table solution, Rec SDK provides essential functions such as feature saving and loading, feature admission, and feature eviction.
3. Large-scale sparse table functions: Rec SDK supports multi-level storage across accelerator card memory, host memory, and host drive. It also supports multi-node storage and dynamic scaling, with capacities exceeding 10 TB.

## Directory Structure

```text
RecSDK/                                          # Project root directory
    |-- build/                                   # Build scripts, generated wheel packages, and more
    |
    |-- cust_op/
    |    |-- ascendc_op                          # Operators written in Ascend C and executed on the AI Core after compilation, along with their build scripts
    |    |-- framework                           # Operator adaptation layer
    |    |-- hkv                                 # HKV submodule code from https://gitcode.com/Ascend/HierarchicalKV-ascend.git
    |    |-- test                                # Operator test cases
    |    |-- tf_cpu_op                           # CPU operators
    |
    |-- docs/                                    # Project documentation, image build scripts, public network/email addresses, and communication matrix documents
    |
    |-- training
         |-- common                              # Common components
         |-- tf_rec_v1                           # Non-fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices
         |-- tf_rec_v2                           # Fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices (PoC)
         |-- torch_rec_v1                        # Non-fully-offloaded sparse recommendation framework based on open-source PyTorch and TorchRec, adapted for NPU devices
         |-- torch_rec_v2                        # Fully-offloaded sparse recommendation framework based on open-source PyTorch and TorchRec, adapted for NPU devices (PoC)
```

### Component Description

| Component| Base Framework| Adaptation Status| Framework Type| Description|
|---------|---------|---------|---------|---------|
| tf_rec_v1 | TensorFlow | Non-fully-offloaded| Sparse recommendation framework| Non-fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices|
| tf_rec_v2 | TensorFlow | Fully-offloaded| Sparse recommendation framework| Fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices (PoC)|
| torch_rec_v1 | PyTorch + TorchRec | Non-fully-offloaded| Sparse recommendation framework| Non-fully-offloaded sparse recommendation framework based on open-source PyTorch and TorchRec, adapted for NPU devices|
| torch_rec_v2 | PyTorch + TorchRec | Fully-offloaded| Sparse recommendation framework| Fully-offloaded sparse recommendation framework based on open-source PyTorch and TorchRec, adapted for NPU devices (PoC)|

### Key Terminology

- **Non-fully-offloaded**: A hybrid mode in which some computing tasks are executed on the NPU and others on the CPU
- **Fully-offloaded**: A mode where all computing tasks are offloaded to the NPU for better performance
- **PoC**: Proof of concept, indicating that the component is still in the experimental verification phase and features may be incomplete or unstable

## Versioning

Typically, Rec SDK has four official release versions per year.

For details about the version updates, see:

* [releases](https://gitcode.com/Ascend/RecSDK/releases)

## Environment Deployment

The following product models are supported:

* Atlas 200T A2 Box16
* Atlas 800T A2 training server
* Atlas 900 A3 SuperPoD

For details about the component deployment methods, see:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md)
* [torch_rec_v1](./docs/en/torch/torch_rec_v1/recsdk_torch_installation_guide.md)
* [torch_rec_v2](./docs/en/torch/torch_rec_v2/recsdk_torch_installation_guide.md)

## Installation from Source

Refer to specific components:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md#installing-from-source)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md#installing-from-source)
* [torch_rec_v1](./docs/en/torch/torch_rec_v1/recsdk_torch_installation_guide.md#installing-from-source)
* [torch_rec_v2](./docs/en/torch/torch_rec_v2/recsdk_torch_installation_guide.md#installing-from-source)

## Quick Start

Refer to specific components:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/quick_start.md)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/quick_start.md)
* [torch_rec_v1](./docs/en/torch/torch_rec_v1/quick_start.md)
* [torch_rec_v2](./docs/en/torch/torch_rec_v2/quick_start.md)

## Features

Refer to specific components:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/introduction.md)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/introduction.md)
* [torch_rec_v1](./docs/en/torch/torch_rec_v1/introduction.md)
* [torch_rec_v2](./docs/en/torch/torch_rec_v2/introduction.md)

## API Reference

Refer to specific components:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/api)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/api)
* [torch_rec_v1](./docs/en/torch/torch_rec_v1/api)
* [torch_rec_v2](./docs/en/torch/torch_rec_v2/api)

## Mode Adaptation Examples

| Model| Framework| Component| Description|
|---------|---------|---------|------|
| DIN | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/din)|
| DLRM(DCNv2) | PyTorch | torch_rec_v1 | [Code link](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/dlrm/README.md)|
| GR | PyTorch | torch_rec_v1 | Facebook GR model: [code Link](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/gr/gr_meta)|
| GR | PyTorch | torch_rec_v1 | NVIDIA recsys-GR model: [code Link](https://gitcode.com/Ascend/RecSDK/tree/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/gr_nv)|
| MMOE, ETA| PyTorch | torch_rec_v1 | [Code Link](https://gitcode.com/Ascend/RecSDK/blob/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/model_zoo/README.md)|
| GR | PyTorch | torch_rec_v2 | NVIDIA recsys-GR model: [Code Link](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/gr)|

## Branch Maintenance Strategy

| Branch| Description| Maintenance Status|
|---------|---------|---------|
| develop | Main development branch| Long-term maintenance|
| develop_examples_and_tools | Demos, model examples, and model-related tools| Active maintenance|
| develop_torch_benchmark | Benchmark model adaptation| Active maintenance|

## FAQ

Refer to specific components:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/faq.md)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/faq.md)

## How to Contribute

Before contributing, sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1).

1. If you encounter a bug, submit an [issue](https://gitcode.com/Ascend/RecSDK/issues).
2. If you plan to contribute bug fixes, submit a pull request (PR). See [Contribution Requirements](./contributing.md#pullrequest).
3. If you plan to contribute new features or functionality, create an issue to discuss it with us first. Describe the background or purpose of the requirement, the design, and its impact on existing APIs. Submitting a PR without prior discussion may lead to rejection, as the evolution direction of the project might differ from your ideas.

## Contact Us

For more detailed information on communication and contribution, see [Contribution Guidelines](./contributing.md).

## Security Statement

You should re-evaluate the network security posture of the entire system based on specific service requirements. When necessary, consult industry best practices and security experts.

For specific security hardening measures, see specific components:

* [tf_rec_v1](./docs/en/tensorflow/tf_rec_v1/security_hardening.md)
* [tf_rec_v2](./docs/en/tensorflow/tf_rec_v2/security_hardening.md)
* [torch_rec_v1](./docs/en/torch/torch_rec_v1/security_hardening.md)
* [torch_rec_v2](./docs/en/torch/torch_rec_v2/security_hardening.md)

## Disclaimer

This repository contains multiple development branches, which may include unfinished, experimental, or untested features. These branches should not be used in any production environment or service-critical projects before an official release. Ensure you use our official release versions to guarantee stability and security.
This project and its contributors are not responsible for any issues, losses, or data corruption resulting from the use of development branches.
For official versions, see the [release](https://gitcode.com/Ascend/RecSDK/releases) page.

## License

Apache License Version 2.0. See the [LICENSE file](./LICENSE) for details.
Documents in the `docs` directory of the Rec SDK are licensed under CC-BY 4.0. For details, see the [LICENSE file](./docs/LICENSE).

## Acknowledgments

The Rec SDK is jointly developed by the following Huawei departments:

* Ascend Computing Application Enablement Development Dept
* Software Platform Dept, Computing Product Line
* UnifiedBus Computing Cluster Development Dept
* Technology Development Dept, Computing Product Line
* Poisson Lab

Thank you to everyone in the community for your PRs. We warmly welcome your contributions to Rec SDK.
