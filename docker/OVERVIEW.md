# RecSDK Docker Image Build Overview

> English | [中文](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/OVERVIEW.zh.md)

This document is compiled from the Dockerfiles provided in the `docker` directory, aiming to help developers quickly understand and use these container images.

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
- **OS version**: Base operating system codename or version number (e.g., `ubuntu20.04`)
- **Framework identifier**: The ML framework supported by the image (e.g., `tf` for TensorFlow, or `torch` for PyTorch)
- **Python version**: The core interpreter version (e.g., `py3.7`)

## Dockerfile Archive Paths

The project currently supports container scenarios divided by underlying frameworks. The corresponding build scripts are located at the following paths:

> [!NOTE]
> Dockerfiles will be published simultaneously with the Atlas community and Atlas image registry. They are not yet updated at this time.

### RecSDK 26.1.0

| Tag | Dockerfile |
|-----|------------|
|26.1.0-ubuntu20.04-tf-py3.7|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-ubuntu20.04-tf-py3.7)|
|26.1.0-ubuntu22.04-pt-py3.11|[Dockerfile](https://gitcode.com/Ascend/RecSDK/blob/develop/docker/Dockerfile.26.1.0-ubuntu22.04-pt-py3.11)|

## Quick Start

### Running an Existing Image

If you have already pulled or built a usable image, you can quickly start and enter a shell environment with the following command:

```bash
docker run -it \
    --name {container_name} \
    --net=host \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v {mount_dir}:{mount_dir} \
    {image_name}:{image_tag} \
    /bin/bash
```

Parameter descriptions:

- ASCEND_VISIBLE_DEVICES=0-7 — The current machine has NPU cards 0-7. For 16 cards, set to 0-15. Adjust based on your actual configuration.
- -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ — The driver directory mounted into the container. Configure according to your actual driver installation path.
- -v /etc/ascend_install.info:/etc/ascend_install.info — The container needs to mount the driver/firmware installation info.
- -m 300g — Set the available memory in the container. Configure based on your usage.

### Building Images Locally

Developers can locally package and build business images in the current system environment. Before doing so, ensure the corresponding packaged artifacts exist in the outer directory (e.g., `output/tf_rec_v1_*.tar.gz`, etc.), as the Dockerfile uses `COPY` to include them.

Build a TensorFlow-based image:

```bash
docker build -t recsdk_tf:26.1.0-ubuntu20.04-tf-py3.7 -f docker/Dockerfile.26.1.0-ubuntu20.04-tf-py3.7 .
```

If you need to target a specific core type (default is `a2`) when compiling operators or loading framework packages, you can explicitly specify it via the `CORE_TYPE` build argument:

```bash
docker build --build-arg CORE_TYPE=a2 -t recsdk_tf_a2:26.1.0-ubuntu20.04-tf-py3.7 -f Dockerfile.26.1.0-ubuntu20.04-tf-py3.7 .
```

### Secondary Development and Underlying Environment Switching

The image provides convenient environment switching scripts for development and debugging work that requires switching between different underlying dependency frameworks or target chip types:

- **How to perform secondary development**:

  ```dockerfile
  # Use recsdk_tf:26.1.0-ubuntu20.04-tf-py3.7 as the base image and layer user software on top
  FROM recsdk_tf:26.1.0-ubuntu20.04-tf-py3.7
  RUN apt update -y &&
      apt install ...
  ```

- **Switch Python framework environment (TensorFlow container)**:

  ```bash
  # Activate the RecSDK development environment for TF 1.15.0
  source /opt/buildtools/tf1_env/bin/activate
  # Exit the environment
  deactivate tf1_env

  # Activate the RecSDK development environment for TF 2.6.5
  source /opt/buildtools/tf2_env/bin/activate
  # Exit the environment
  deactivate tf2_env
  ```

- **Switch Python framework environment (PyTorch container)**:

  ```bash
  # Activate the RecSDK development environment for torch_rec_v1 PT 2.6.0
  source /opt/buildtools/torch_v1_pt2.6.0/bin/activate
  # Exit the environment
  deactivate torch_v1_pt2.6.0

  # Activate the RecSDK development environment for torch_rec_v1 PT 2.7.1
  source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
  # Exit the environment
  deactivate torch_v1_pt2.7.1

  # Activate the RecSDK development environment for torch_rec_v2 PT 2.7.1
  source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
  # Exit the environment
  deactivate torch_v2_pt2.7.1
  ```

- **Switch CANN hardware support packages**:

  Use the built-in switching script to dynamically adjust the environment variables and symlink paths bound to CANN packages:

  ```bash
  source /usr/local/set_cann_env.sh a2  # Switch to Atlas 800T2 training server toolkit and related env vars (system default)
  source /usr/local/set_cann_env.sh a3  # Switch to Atlas 800T3 super node server toolkit and related env vars
  source /usr/local/set_cann_env.sh a5  # Switch to Ascend 950 toolkit and related env vars
  ```

## Support Information and Changelog

### Hardware Support Information

- **Automatic physical architecture adaptation**: Dockerfiles across all versions natively support host architecture detection. They not only cover `x86` and `ARM` hardware handling, but also identify the architecture early in the build process via `ARCH=$(uname -m)`, fetching and assembling the corresponding OS-specific `GCC` configuration, system dynamic libraries, and framework distribution versions (i.e., different `whl` packages and `run` packages) based on logical branching.
- **Full chip compatibility deployment**: The build process avoids limitations tied to a single accelerator card or compute board. By leveraging pre-installed multi-version compute deployment solutions, the corresponding compute operation stacks (ops packages and toolkit packages) for A2/A3/A5 are all deposited and installed into the image base, greatly enhancing the container's distribution scalability and business versatility.

### Compatibility Changes

- Abandoned the practice of mixing multiple highly-exclusive framework modules under the global system Python, which previously led to dependency disasters (such as the common `npu_bridge` and `npu_device` collisions and Slim compatibility issues). The new scripts universally adopt `python3.7 -m venv` for virtual-environment-isolated deployment, ensuring each framework module runs independently without mutual awareness or interference.
- For dependency suites highly coupled with hardware logic operations (including important runtime libraries such as GCC 11.2, CMake, OpenMPI, Python, etc.), the system package manager distributions are no longer used. Instead, they are built from source using `make install`.

## License / Disclaimer

- View the [license information](https://gitcode.com/Ascend/RecSDK/blob/develop/LICENSE) for RecSDK and Mind series software included in these images.
- As with all container images, pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
