# RecSDK Image Creation (Using a Dockerfile)

## Prerequisites

- The driver and firmware of the target CANN version have been installed on the physical machine.
- Docker has been installed on the physical machine and the Docker network is available.
- A base OS image has been prepared on the physical machine by running one of the following commands:
  - Debian, x86 architecture: `docker pull debian:12` (pulls the image from Docker Hub)

## Build Steps

Step 1: Create a `build_images` directory.

Step 2: Prepare the CANN packages in the `build_images` directory. You can download the **9.0.0** toolkit package and operator package from the [Ascend community](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26). You can also choose another CANN package version as required.

Starting with CANN 8.5.0, the operator package name changed. When you download the operator package, make sure that you select the package with the correct name.

| Version/Package Name     | Before CANN 8.5.0                                | CANN 8.5.0 and Later                                         |
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

Step 3: Move the Dockerfile to the `build_images` directory, and then run the following command to build the image. The Dockerfile contains detailed instructions for the build process. The commented sections cover installation of the CANN packages and the torchrec-related packages.

```shell
# If the server can directly access the Internet:
docker build -t recsdk_torch_base:v2.0-[x86|arm] -f Dockerfile_debian .
# If the server can access the Internet through a proxy:
docker build -t recsdk_torch_base:v2.0-[x86|arm] -f Dockerfile_debian --build-arg http_proxy=http://your_proxy --build-arg https_proxy=https://your_proxy .
```

**Note:**

1. Make sure that the server can access the internet when you build the image. Otherwise, configure a proxy.
2. Before you run the command, modify the base image name in the Dockerfile.
