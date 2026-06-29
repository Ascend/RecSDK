
# RecSDK Image Creation (Using a Dockerfile)

## Prerequisites

- The driver and firmware of the target CANN version have been installed on the physical machine.
- Docker has been installed on the physical machine and the Docker network is available.
- A base OS image has been prepared on the physical machine by running one of the following commands:
  - Debian, x86 architecture: `docker pull debian:12`
  - openEuler, Arm architecture: `wget https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/docker_img/aarch64/openEuler-docker.aarch64.tar.xz && docker load -i openEuler-docker.aarch64.tar.xz`<br>
  - CentOS, x86 architecture: `docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/centos:7.6.1810`<br>

## Version Mapping

Rec SDK Torch currently supports the following version combinations:

| Version Combination | PyTorch | torch_npu | fbgemm_gpu |
|-------|---------|-----------|------------|
| Version combination 1| 2.6.0   | 2.6.0     | 1.1.0      |
| Version combination 2| 2.7.1   | 2.7.1     | 1.2.0      |

The current Dockerfile downloads the software packages for PyTorch 2.6.0 and fbgemm_gpu 1.1.0+cpu by default. You can modify the corresponding download commands in the Dockerfile as needed.

If you need to build an image for the PyTorch 2.7.1 version combination, see [Building a PyTorch 2.7.1 Image](#building-a-pytorch-271-image).

## Build Steps

Step 1: Create a `build_images` directory.

Step 2: Prepare the CANN packages in the `build_images` directory. You can download the 8.5.0 [toolkit package](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0) and [operator package](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0) from the [Ascend community](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26). You can also choose another CANN package version as required.

Starting with CANN 8.5.0, the operator package name changed. When you download the operator package, make sure that you select the package with the correct name.

| Version/Package Name     | Before CANN 8.5.0                                  | CANN 8.5.0 and Later                                         |
|------------|------------------------------------------------|--------------------------------------------------------|
| Toolkit package name| Ascend-cann-toolkit_{version}_linux-{arch}.run | Ascend-cann-toolkit_{version}_linux-{arch}.run         |
| Operator package name     | Ascend-cann-kernels-{version}_linux-{arch}.run | Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run |

> Note
>
> 1. The current Dockerfile uses the operator package name for CANN 8.5.0 and later.
> 2. To install a CANN package earlier than 8.5.0, change the value of `KERNEL_PKG` in the Dockerfile to `Ascend-cann-kernels*.run`.

CANN package installation requires the following two files:

- `version.info` (driver version file)
- `ascend_install.info` (firmware and driver installation file)

Copy the corresponding files from the physical machine to the `build_images` directory. After you install the driver and firmware on the physical machine, the default path of `version.info` is `/usr/local/Ascend/driver/version.info`.
The default path of `ascend_install.info` is `/etc/ascend_install.info`.

Step 3: Move [Dockerfile_centos](./Dockerfile_centos), [Dockerfile_debian](./Dockerfile_debian), or [Dockerfile_openeuler](./Dockerfile_openeuler) to the `build_images` directory based on the base OS image, and then run the following command to build the image. The Dockerfile contains detailed instructions for the build process. The commented sections cover installation of the CANN packages and the torchrec-related packages.

```shell
# If the server can directly access the Internet:
docker build -t recsdk_torch_base:v1.0-[x86|arm] -f Dockerfile_[centos|debian|openeuler] .
# If the server can access the Internet through a proxy:
docker build -t recsdk_torch_base:v1.0-[x86|arm] -f Dockerfile_[centos|debian|openeuler] --build-arg http_proxy=http://your_proxy --build-arg https_proxy=https://your_proxy .
```

**Note:**

1. Make sure that the server can access the internet when you build the image. Otherwise, configure a proxy.
2. Before you run the command, modify the base image name in the Dockerfile.

## Starting the Container

Create startup script `run_docker.sh`, as shown in the following example:

```shell
#!/bin/bash
container_name=$1
image_name=$2
docker run \
-it \
--name "${container_name}" \
--shm-size="300g" \
-m 300g  \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```

Description of some parameters:

- -`-m 300g`: Sets the amount of memory available in the container. Adjust it as required.
- -`-e ASCEND_VISIBLE_DEVICES=0-7`: Mounts NPU devices `device0` to `device7` from the server into the container. Adjust it as required.

Run the following command to create a container:

```shell
bash run_docker.sh <container_name> {image_name}:{version_name}
```

## Installing RecSDK-Related Packages

For source compilation and installation, see [Installation and Deployment](../torch_rec_v1/recsdk_torch_installation_guide.md#installation-and-deployment).

## Building a PyTorch 2.7.1 Image

You can use either of the following methods to build an image environment for PyTorch 2.7.1.

### Method 1: Modifying the Dockerfile and Rebuild the Image

Modify an existing Dockerfile and then rebuild the image.

1. Update the download links for PyTorch and fbgemm_gpu. The download links are as follows:

    | Software Version                  | Python Version| Download Link                                                                                              |
    |------------------------|------------|----------------------------------------------------------------------------------------------------|
    | PyTorch 2.7.1 (x86)    | 3.11       | <https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl>       |
    | PyTorch 2.7.1 (Arm)    | 3.11       | <https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl>      |
    | fbgemm_gpu 1.2.0 (x86) | 3.11       | <https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.2.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl>  |
    | fbgemm_gpu 1.2.0 (Arm) | 3.11       | <https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.2.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl> |

2. Update the package names that you use when installing PyTorch, fbgemm_gpu, and torch_npu.

    | Before Modification                  | After Modification                       |
    |----------------------------|----------------------------|
    | torch-2.6.0+cpu-*.whl      | torch-2.7.1+cpu-*.whl      |
    | fbgemm_gpu-1.1.0+cpu-*.whl | fbgemm_gpu-1.2.0+cpu-*.whl |
    | torch-npu==2.6.0           | torch-npu==2.7.1           |

3. To build the image, see [Build Steps](#build-steps).

### Method 2: Upgrading the Software Versions in an Existing Image

Upgrade the PyTorch, fbgemm_gpu, and torch_npu versions in an existing image to the versions that correspond to PyTorch 2.7.1.

1. Upgrade PyTorch.

    ```shell
    pip3 install torch==2.7.1+cpu -i https://download.pytorch.org/whl/cpu
    ```

2. Upgrade fbgemm_gpu.

    ```shell
    pip3 install fbgemm_gpu==1.2.0+cpu -i https://download.pytorch.org/whl/cpu
    ```

3. Upgrade torch_npu.

    ```shell
    pip3 install torch-npu==2.7.1
    ```
