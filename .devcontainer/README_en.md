# RecSDK Dev Container Guide

This directory contains the VS Code Dev Container configuration for building and debugging RecSDK inside a container.

## Environment Specifications

| Software      | Version       |
| ------------- | ------------- |
| Base Image    | Debian 12     |
| PyTorch       | 2.6.0         |
| Python        | 3.11.0        |
| fbgemm-gpu    | 1.1.0         |
| GCC           | 11.2.0        |
| CMake         | 3.22.6        |
| CANN          | 9.0.0-beta.2  |
| torch_npu     | 7.1.0         |
| RecSDK-torch  | 26.0.0        |

## Quick Start

### 1. Start the Container

Open the repository in VS Code, press `F1` → **Dev Containers: Reopen in Container**, and wait for the image to be pulled.

### 2. Build RecSDK

The torch_rec_v1 packages are pre-installed by default. For source build and installation, refer to the [Source Installation Guide](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/zh/torch/torch_rec_v1/recsdk_torch_installation_guide.md#%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85).

## Notes

- The container runs as **root** user, matching the permission requirements of the CANN toolkit installation path `/usr/local/Ascend`
- Full compilation and execution require **NPU driver** and **CANN toolkit** (the toolkit is pre-installed in the image; the driver must be provided by the host)
- The workspace directory `${localWorkspaceFolder}` is mounted into the container at `/workspace` via bind mount, and changes are synchronized in real time
