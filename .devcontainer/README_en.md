# RecSDK Dev Container Guide

This directory contains the VS Code Dev Container configuration for building and debugging RecSDK inside a container.

## Environment Specifications

| Software      | Version       |
| ------------- | ------------- |
| Base Image    | ubuntu22.04   |
| PyTorch       | 2.7.1         |
| Python        | 3.11.0        |
| fbgemm-gpu    | 1.2.0         |
| GCC           | 11.2.0        |
| CMake         | 3.22.6        |
| CANN          | 9.1.0         |
| torch_npu     | 2.7.1         |
| RecSDK-torch  | 26.1.0        |

## Quick Start

### 1. Start the Container

Open the repository in VS Code, press `F1` → **Dev Containers: Reopen in Container**, and wait for the image to be pulled.

### 2. Set Up Environment Variables

```shell
# Set CANN environment variables
source /usr/local/Ascend/cann/set_env.sh
# Activate the Python virtual environment if it exists. To deactivate it later, run: deactivate
[ -f /opt/buildtools/torch_v1_pt2.7.1/bin/activate ] && source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
```

### 3. Build RecSDK

The torch_rec_v1 packages are pre-installed by default. For source build and installation, refer to the [Source Installation Guide](https://gitcode.com/Ascend/RecSDK/blob/develop/docs/en/torch/torch_rec_v1/recsdk_torch_installation_guide.md#installing-from-source).

## Notes

- The container runs as **root** user, matching the permission requirements of the CANN toolkit installation path `/usr/local/Ascend`
- Full compilation and execution require **NPU driver** and **CANN toolkit** (the toolkit is pre-installed in the image; the driver must be provided by the host)
- The workspace directory `${localWorkspaceFolder}` is mounted into the container at `/workspace` via bind mount, and changes are synchronized in real time
