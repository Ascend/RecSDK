# RecSDK Docker Image Build Overview

> English | [中文](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/OVERVIEW.zh.md)

This document is compiled from the Dockerfiles provided in the [`docker`](https://gitcode.com/Ascend/RecSDK/blob/develop/docker) directory, aiming to help developers quickly understand and use these container images.

## Quick Reference

- Where to get help
  - [Issue feedback](https://gitcode.com/Ascend/RecSDK/issues)
  - [RecSDK source code](https://gitcode.com/Ascend/RecSDK)
  - [RecSDK documentation](https://gitcode.com/Ascend/RecSDK/tree/develop/docs)

## RecSDK Overview

RecSDK is an application-enablement SDK product targeting search, recommendation, and advertising (SRA) scenarios in the Internet market. For the application requirements of search/recommendation/advertising model training, it provides an Atlas-based SRA framework that supports large-scale SRA scenarios and facilitates efficient training of SRA models.

RecSDK features include:

1. Basic model training capabilities. Supports single-node single-device training and multi-node multi-device distributed training.
2. Recommendation-specific features. Based on RecSDK's sparse table solution, RecSDK provides essential functionality such as feature saving/loading, feature admission, and feature eviction.
3. Large-scale sparse table features. Supports multi-level storage across accelerator memory, host memory, and host disk; supports multi-node storage; supports dynamic capacity expansion. Scale can exceed 10 TB.

## Key Fields in Image Tags

Image tags follow the convention below for intuitive identification of the software and hardware stack versions contained within the image:

- **RecSDK version**: RecSDK version number (e.g., `26.1.0`)
- **CANN version**: CANN package version (e.g., `cann9.1.0`)
- **Chip identifier**: Target Atlas chip platform (`910` for Atlas 800T A2 / a2, `a3` for Atlas 800T A3, `950` for Atlas 950 / a5)
- **OS version**: Base operating system codename or version number (e.g., `ubuntu20.04`)
- **Python version**: The core interpreter version (e.g., `py3.7`)
- **Framework identifier**: The ML framework supported by the image (e.g., `tf` for TensorFlow, or `pt` for PyTorch)

> **Note**: The same Dockerfile can produce images for different chip platforms via `--build-arg CORE_TYPE=a2/a3/a5`. Replace the chip identifier field in the tag according to the target platform:
>
> - `CORE_TYPE=a2` → use `910` in tag
> - `CORE_TYPE=a3` → use `a3` in tag
> - `CORE_TYPE=a5` → use `950` in tag

## Dockerfile Archive Paths

The project currently supports container scenarios divided by underlying frameworks. The corresponding build scripts are located at the following paths:

### RecSDK 26.1.0

| Tag | Dockerfile |
|-----|------------|
|26.1.0-cann9.1.0-{chip}-ubuntu20.04-py3.7-tf|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf)|
|26.1.0-cann9.1.0-{chip}-ubuntu22.04-py3.11-pt|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-ubuntu22.04-py3.11-pt)|
|26.1.0-cann9.1.0-{chip}-openEuler22.03-py3.7-tf|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-openEuler22.03-py3.7-tf)|
|26.1.0-cann9.1.0-{chip}-openEuler22.03-py3.11-pt|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-cann9.1.0-openEuler22.03-py3.11-pt)|

> Replace `{chip}` with the target chip identifier: `910` (Atlas 800T A2 / a2), `a3` (Atlas 800T A3), `950` (Atlas 950 / a5).

## Quick Start

### Prerequisites

| Item | Requirement |
|------|-------------|
| Docker Version | 20.10 or higher recommended; must support `--net=host` network mode |
| Host OS | Ubuntu 20.04 / 22.04 (x86_64 or ARM), openEuler 22.03 (x86_64 or ARM) |
| Atlas Driver & Firmware | Host must have Atlas NPU driver and firmware installed; default driver path is `/usr/local/Ascend/driver` |
| CANN Version | CANN 9.1.0 or higher recommended (Dockerfile can be configured with a download URL; defaults to version 9.1.0) |
| Disk Space | At least 60 GB of free space recommended for image builds |
| Network | External network access is required during the build process to download dependency packages |

### Running an Existing Image

If you have already pulled or built a usable image, you can quickly start and enter a shell environment with the following command:

```bash
docker run -it \
    --name {container_name} \
    --net=host \
    -m 300g \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v {mount_dir}:{mount_dir} \
    {image_name}:{image_tag} \
    /bin/bash
```

Parameter descriptions:

- ASCEND_VISIBLE_DEVICES=0-7 — The current machine has NPU cards 0-7. For 16 cards, set to 0-15. Adjust based on your actual configuration.
- -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro — The driver directory mounted into the container (read-only). Configure according to your actual driver installation path.
- -v /etc/ascend_install.info:/etc/ascend_install.info — The container needs to mount the driver/firmware installation info.
- -m 300g — Set the available memory in the container. Configure based on your usage.

### Building Images Locally

Developers can locally package and build business images in the current system environment. The Dockerfiles already include the download and installation logic for software packages, so no external resources need to be prepared in advance. You may modify the software package versions in the Dockerfile as needed (CANN package version and RecSDK package version).

- **PyTorch Image (using ubuntu22.04 as an example)**

```bash
# Specify chip platform via --build-arg CORE_TYPE, replace {chip} in tag with 910/a3/950
docker build --build-arg CORE_TYPE=a2 -t recsdk_pt:26.1.0-cann9.1.0-910-ubuntu22.04-py3.11-pt -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu22.04-py3.11-pt .
```

- **TensorFlow Image (using ubuntu20.04 as an example)**

```bash
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf:26.1.0-cann9.1.0-910-ubuntu20.04-py3.7-tf -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf .
```

If you need to target a specific core type when compiling operators or loading framework packages, you can explicitly specify it via the `CORE_TYPE` build argument (default is `a2`):

| CORE_TYPE | Applicable Platform | Tag Chip Identifier |
|-----------|---------------------|---------------------|
| `a2` | Atlas 800T A2 Training Server | `910` |
| `a3` | Atlas 800T A3 Super Node Server | `a3` |
| `a5` | Atlas 950 Generation | `950` |

```bash
# CORE_TYPE can be a2/a3/a5. The following uses TensorFlow as an example; the same approach applies to PyTorch images.
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf:26.1.0-cann9.1.0-910-ubuntu20.04-py3.7-tf -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf .
```

- **openEuler Images**

The build process for openEuler 22.03 images follows the same approach as the Ubuntu examples above.

### Secondary Development and Underlying Environment Switching

The image provides convenient environment switching scripts for development and debugging work that requires switching between different underlying dependency frameworks or target chip types:

- **How to perform secondary development**:

  ```dockerfile
  # Use recsdk_tf:26.1.0-cann9.1.0-910-ubuntu20.04-py3.7-tf as the base image and overlay user software
  FROM recsdk_tf:26.1.0-cann9.1.0-910-ubuntu20.04-py3.7-tf
  RUN apt update -y && \
      apt install ...
  ```

- **Switch Python framework environment (TensorFlow container)**:

  ```bash
  # Activate the RecSDK development environment for TF 1.15.0
  source /opt/buildtools/tf1_env/bin/activate

  # Activate the RecSDK development environment for TF 2.6.5
  source /opt/buildtools/tf2_env/bin/activate
  ```

- **Switch Python framework environment (PyTorch container)**:

  ```bash
  # Activate the RecSDK development environment for torch_rec_v1 PT 2.6.0
  source /opt/buildtools/torch_v1_pt2.6.0/bin/activate

  # Activate the RecSDK development environment for torch_rec_v1 PT 2.7.1
  source /opt/buildtools/torch_v1_pt2.7.1/bin/activate

  # Activate the RecSDK development environment for torch_rec_v2 PT 2.7.1
  source /opt/buildtools/torch_v2_pt2.7.1/bin/activate

  # Activate the RecSDK development environment for torch_rec_v2 PT 2.10.0
  source /opt/buildtools/torch_v2_pt2.10.0/bin/activate
  ```

  Use `deactivate` to exit the current virtual environment.

## Support Information and Changelog

### Hardware Support Information

- **Automatic physical architecture adaptation**: Dockerfiles across all versions natively support host architecture detection. They not only cover `x86` and `ARM` hardware handling, but also identify the architecture early in the build process via `ARCH=$(uname -m)`, fetching and assembling the corresponding OS-specific `GCC` configuration, system dynamic libraries, and framework distribution versions (i.e., different `whl` packages and `run` packages) based on logical branching.
- **On-demand chip build**: The build process specifies the target chip platform via `--build-arg CORE_TYPE=a2/a3/a5`, installing only the CANN toolkit and ops packages for the corresponding chip architecture, avoiding image bloat.

### Compatibility Changes

- Abandoned the practice of mixing multiple framework modules under the global system Python to avoid dependency conflicts between different modules. The new version uses `python3.7 -m venv` to create an independent virtual environment for each framework, ensuring each framework module runs in isolation without interfering with one another.
- For dependency suites highly coupled with hardware logic operations (including important runtime libraries such as GCC 11.2, CMake, OpenMPI, Python, etc.), the system package manager distributions are no longer used. Instead, they are built from source using `make install`.

## Quick Verification

### End-to-End Verification Flow Example

The following uses the TensorFlow image as an example to demonstrate the complete workflow from build to verification:

```bash
# 1. Build the image
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf:26.1.0-cann9.1.0-910-ubuntu20.04-py3.7-tf \
  -f docker/Dockerfile.26.1.0-cann9.1.0-ubuntu20.04-py3.7-tf .

# 2. Start the container
docker run -it --name recsdk_test \
  --net=host \
  -m 300g \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  recsdk_tf:26.1.0-cann9.1.0-910-ubuntu20.04-py3.7-tf /bin/bash

# 3. Verify NPU availability inside the container
npu-smi info

# 4. Activate the environment and run the little demo
source /opt/buildtools/tf1_env/bin/activate
# Refer to the little demo link below to run verification
```

### TensorFlow Image Verification

Refer to the [little demo](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/examples/demo/README.md) for verification. Before running, switch the Python virtual environment via [Secondary Development and Underlying Environment Switching](#secondary-development-and-underlying-environment-switching).

### PyTorch Image Verification

Refer to the [little demo](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/little_demo/README.md) for verification. Before running, switch the Python virtual environment via [Secondary Development and Underlying Environment Switching](#secondary-development-and-underlying-environment-switching).

## FAQ and Troubleshooting

> [!NOTE]
> If you encounter other issues, please submit feedback at [RecSDK Issues](https://gitcode.com/Ascend/RecSDK/issues).

### NPU Not Visible in Container

**Symptom**: Running `npu-smi info` inside the container produces no output or an error.

**Cause**:

- The host does not have the NPU driver installed, or the driver version is incompatible.
- The driver directory `/usr/local/Ascend/driver` is not mounted.
- `ASCEND_VISIBLE_DEVICES` is not set, or is set to a device number that does not exist.

**Resolution**:

1. Run `npu-smi info` on the host to confirm the driver is working properly.
2. Verify that `docker run` includes `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro` and `-v /etc/ascend_install.info:/etc/ascend_install.info`.
3. Confirm that the value of `ASCEND_VISIBLE_DEVICES` is within the range of available NPU devices on the host.

### Container Memory Insufficient

**Symptom**: Training tasks encounter OOM errors or run slowly.

**Cause**: The memory limit set by the `-m` parameter is too small.

**Resolution**:

1. Increase the `-m` value (e.g., `-m 500g`), or remove this parameter to use all available host memory.
2. Confirm that the host has sufficient free memory.

### Framework Version Not Taking Effect After Environment Switch

**Symptom**: After running `source activate`, `python -c "import torch; print(torch.__version__)"` still shows the old version.

**Cause**: Another Python environment takes precedence in `PATH`.

**Resolution**:

1. Run `deactivate` to exit the current virtual environment first, then `source` the new environment.
2. Use `which python` to confirm the path points to the virtual environment under `/opt/buildtools/`.

## License / Disclaimer

- View the [license information](https://gitcode.com/Ascend/RecSDK/blob/develop/LICENSE) for RecSDK and Mind series software included in these images.
- As with all container images, pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
