# Description

This document describes how to create Rec SDK training images based on existing images.

## Document Structure

```shell
└── build_mxRec_images
    ├── centos_build    # Uses open-source CentOS images from AscendHub or customer's own images as base images.
    │   └── Dockerfile
    ├── mxrec-build     # Uses open-source Rec SDK images from AscendHub as base images.
    │   └── Dockerfile
    └── README.md       # Description
```

## Prerequisites

The driver and firmware of the target CANN version have been installed on the physical machine.

Docker has been installed on the physical machine and the Docker network is available.

Prepare a base image. If you have not prepared a base image, you can pull one from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub/). You are advised to pull the following images:

* Preferentially pull the Rec SDK training image. The Rec SDK training image on AscendHub already has basic dependencies such as GCC and CMake installed, eliminating the need for manual installation.
Additionally, the image includes CANN and Rec SDK packages, though with older versions. Therefore, when using the Rec SDK image as the base image, you only need to update the CANN and Rec SDK packages.
* Alternatively, pull the [CentOS 7.6.1810](https://www.hiascend.com/developer/ascendhub/detail/9353d9619c2a44db87845bce546c17bd) image from AscendHub.
* Finally, if neither of the preceding images is used, you can prepare your own image as a base image. It is recommended that this image be based on CentOS 7.6.1810.

## Dependency Preparation

Depending on the base image, the dependencies to be downloaded vary.

1. When using the Rec SDK training image from AscendHub as the base image, you only need to download the latest CANN, tf-plugin, and Rec SDK installation packages of the latest matching version from the [Ascend Community](https://www.hiascend.com/developer/download/community/result?module=sdk+cann). Download the CANN and Rec SDK of the matching version from the following links:

    <https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann>

    <https://www.hiascend.com/developer/download/community/result?module=tf+cann>

    For specific image building steps, refer to the Dockerfile in the `mxrec-build` directory.

2. Using CentOS7.6.1810 or a user-provided image as the base image requires more dependencies. You need to confirm whether the following dependencies are already installed in your image. Since many dependencies need to be installed, you are advised to follow the steps in the Dockerfile to **manually install** the dependencies, such as GCC and CMake.

    * gcc-7.3.0

    Download: [https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz](https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)

    * cmake-3.20.6

    Download: [https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz](https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz)

    * ucx

    Download: [https://github.com/openucx/ucx/archive/master.zip](https://github.com/openucx/ucx/archive/master.zip)

    * openmpi-4.1.5

    Download: [https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz)

    * python-3.7.5

    Download: [https://repo.huaweicloud.com/python/3.7.5/Python-3.7.5.tar.xz](https://repo.huaweicloud.com/python/3.7.5/Python-3.7.5.tar.xz)

    * hdf5-1.10.5

    Download: [https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz)

    * CANN and Rec SDK

    The Rec SDK version package released on the Ascend community (<https://www.hiascend.com/developer/download/community/result?module=sdk+cann>) matches the CANN version. Therefore, you need to download Rec SDK, CANN, and tf-plugin of the matching version from the community.
    You can download the Rec SDK and CANN of the matching version from the following links:

    <https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann>

    <https://www.hiascend.com/developer/download/community/result?module=tf+cann>

    * Tensorflow (1.15.0/2.6.5)

    The current Rec SDK is developed based on TensorFlow, so TensorFlow needs to be installed in the environment. In x86 environments, it can be installed directly using `pip` or `pip3` commands.
    However, in Arm environments, TensorFlow does not have a corresponding .whl package and cannot be installed directly using `pip` or `pip3` commands. You can download the TensorFlow of the Arm architecture from the following link:

    [https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html)

    * `version.info` and `ascend_install.info` required for CANN installation

    `version.info` (driver version file) and `ascend_install.info` (firmware driver installation parameters) are required for CANN installation. These two files can be copied from the corresponding files on the physical machine to the same directory.
    By default, `version.info` is installed at `/usr/local/Ascend/driver/version.info`, and the `ascend_install.info` file is located at `/etc/ascend_install.info`.

    For specific image building steps, refer to the Dockerfile in the `centos-build` directory.

    **Recommendation**: **Download the preceding dependencies to the same directory** as required for easy handling. In addition, before using the Dockerfile to build the image, carefully review the corresponding
    Dockerfile, as you need to modify the Dockerfile based on the actual environment. The image building steps are detailed in the Dockerfile.
