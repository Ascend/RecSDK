# Installation and Deployment

## Installation Instructions

- Rec SDK TensorFlow supports deployment of a development environment on a physical machine or in a container. You can choose either method based on service requirements.
- You are advised to install and run the software as a common user.

    >[!NOTE]
    >To view the installation history of the Rec SDK TensorFlow, see [Viewing Rec SDK TensorFlow Installation and Uninstallation Records](common_operations.md#viewing-rec-sdk-tensorflow-installation-and-uninstallation-records).

## Installation Prerequisites

Before you install the Rec SDK TensorFlow software package, prepare the following environment dependencies and perform the listed actions. For details, see [Table 1](#table18461141184116).

**Table 1** Rec SDK TensorFlow environment dependencies
<a id="table18461141184116"></a>

|Dependency/Operation|Recommended Version|How to Obtain|
|--|--|--|
|CANN software package and TensorFlow Ascend adaptation plugin|CANN 9.0.0|<li>`Ascend-cann-toolkit_{version}_linux-{arch}.run`:Click [this link](https://www.hiascend.com/developer/download/commercial/result?module=cann), configure the package in **Edit Resource Selection** under supporting resources on the left, filter the required software package, and obtain the package after you confirm the version information.<br>Install it according to the *CANN Software Installation Guide*. </li><li>For the TensorFlow Ascend adaptation plugin, click [this link](https://gitee.com/ascend/tensorflow/releases/tag/tfa_v0.0.44_8.3.RC1). npu_bridge-1.15.0* is compatible with TensorFlow 1.15.0.</li>|
|Ascend hardware product driver and firmware|Ascend HDK 26.0.RC1 and its patch versions|Click [this link](https://www.hiascend.com/developer/download/commercial/result?module=cann), configure the package in **Edit Resource Selection** under supporting resources on the left, filter the required software package, and obtain the package after you confirm the version information. For driver and firmware installation, see the [driver and firmware installation and upgrade guides](https://support.huawei.com/enterprise/en/ascend-computing/ascend-hdk-pid-252764743) that come with the hardware product.|
|Ascend Docker Runtime|MindCluster 7.3.0|See the **Installation** > **Installation Deployment** section of the *MindCluster Cluster Scheduling User Guide* for installation.|
|Configuring the device NIC|-|Refer to the training node configuration example in the parameter network configuration section of the *Ascend Training Solution Networking Guide*, and configure the device IP address of the NPU network port with HCCN_Tool.|
|TensorFlow|TensorFlow 1.15.0|Obtain the source code from the [TensorFlow](https://github.com/tensorflow/tensorflow) repository. In Arm environments, TensorFlow does not provide the corresponding official wheel package. If you need to use it in Arm environments, you can obtain the Arm TensorFlow wheel package from [this link](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html).<br>[!NOTE]<br>If the wheel package fails to be downloaded, copy the link and open it in a new tab page to complete the download.|
|Python 3.7.5|Python 3.7.5|Obtain the dependency software package from the [Python official website](https://www.python.org/).|

>[!NOTE]
>For open-source and third-party software that you integrate, track vulnerabilities and issues in the community and fix them in a timely manner. You can check the [CVE website](https://www.cve.org/) for known vulnerabilities in the corresponding open-source software version, and fix them by upgrading versions or applying patch packages.

## Obtaining the Rec SDK TensorFlow Software Package

Refer to this section to obtain the Rec SDK TensorFlow software and verify the software digital signature.

If you need the Rec SDK TensorFlow source code, you can obtain it from the [source address](https://gitcode.com/Ascend/RecSDK/tree/develop/training/tf_rec_v2). Note that no official version has been released yet, and the source code is for reference only. You can also build it yourself by following the procedure below.

### Installing from Source

Before you build the source code, follow the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu) to install the CANN development kit software package. Then, follow the [TF Adapter Installation Guide](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/900beta1/migration/tfmigr1/tfmigr1_000009.html) to install the TensorFlow framework plugin package for Ascend adaptation.

Build environment dependencies:

- Python3.7.5
- GCC 11.2.0
- CMake 3.20.6

Open-source dependencies:

- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)
- [openmpi 4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz): Install it in the build environment according to the software documentation.
- TensorFlow 1.15: Select the corresponding version as required.

Place the compressed packages of pybind11 (`>= 2.10.3`) and securec (`huaweicloud-sdk-c-obs-3.23.9.zip`) in the same directory as the RecSDK code under the `opensource` subdirectory.

**Compilation method:**

> [!NOTE]
> The `build_wrapper.sh` script is updated with the resource download center. Currently, you are advised to compile and install the Rec SDK .whl package separately.

Go to the RecSDK code directory.

To compile and install the software package, refer to the `build/build_wrapper/tf_rec_v2/build_wrapper.sh` script and run the script to build the software package. After the build is successful, the software package is stored in the `build/output` subdirectory.

```bash
# Build the package.
bash build/build_wrapper/tf_rec_v2/build_wrapper.sh

# Install the package.
pip install build/output/tf_rec_v2*.tar.gz
```

To build and install the Rec SDK .whl package separately (excluding the automatic TensorFlow version identification function and the installation of framework-dependent operators), run the following command:

- `setup.py`: Run `python3.7 setup.py bdist_wheel` to build the wheel package for TensorFlow 1. After the build succeeds, the wheel package is in the `build` subdirectory.

## Deploying the Development Environment on a Physical Machine

>[!NOTE]
>
>- Currently, the physical machine development environment can be deployed on Ubuntu 20.04 and CentOS 7.
>- Do not modify any code files in the build directory except for `run.sh`.

1. Install the CANN software package and TensorFlow Ascend adaptation plugin by referring to the *CANN Software Installation Guide*.
2. Configure environment variables.

    The CANN software provides a process-level environment variable setting script for users to reference in processes to automatically set environment variables. They become invalid automatically after the user process ends.

    You can use the following command in the shell script that starts the program to set the relevant CANN environment variables. You can also run the following command from the CLI. The following example uses the default root installation path `/usr/local/Ascend`:

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

3. Run the following command to install the software package:

    ```bash
    # Install the package.
    pip install tf_rec_v2-{version}-{arch}.tar.gz
    ```

    (`{version}` indicates the version number, and `{arch}` indicates the OS architecture. Replace them with the actual ones.)

    The software package installs in Python `site-packages` by default. If you specify a directory with the `--target` parameter, add the Rec SDK TensorFlow path to the `PYTHONPATH` environment variable after installation.

    ```bash
    export PYTHONPATH={rec_install_path}:{rec_install_path}/tf_rec_v2:$PYTHONPATH
    ```

4. Install dependencies. If you do not build an image and instead develop directly on a physical machine, you must install the following Python dependencies.

    ```bash
    pip3.7 install numpy decorator sympy cffi pyyaml pathlib2 grpcio grpcio-tools protobuf==3.20.3 scipy requests mpi4py easydict scikit-learn attrs toml
    ```

    Before you install the horovod dependency, configure `HOROVOD_WITH_MPI` and `HOROVOD_WITH_TENSORFLOW`. The installation command is as follows:

    ```bash
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip3.7 install horovod --no-cache-dir
    ```

5. If you need the Hadoop distributed file system, refer to the [Hadoop official documentation](https://hadoop.apache.org/docs/r1.0.4/cn/quickstart.html) for environment deployment and cluster setup. Hadoop 2.7.5 is recommended.

## Deploying the Development Environment in a Container

### Using a Container to Deploy the Development Environment

>[!NOTE]
>
>- If you use CentOS for configuration, including the host and the container, the `libstdc++` version must be later than `libstdc++.so.6.0.24`.
>- For security purposes, only non-root users can start and use containers.

You can deploy the Rec SDK TensorFlow development environment in a container by following the steps in [Figure 1](#fig2687191413442).

**Figure 1** Configuring the development environment in a container and building the training image<a id="fig2687191413442"></a>
![](../../figures/tf_rec_v1/configuring-the-development-environment-in-a-container-and-building-the-training-image.png "Configuring the development environment in a container and building the training image")

**Key steps**

1. Prepare the host environment.

    See [Installation Prerequisites](#installation-prerequisites) to deploy the host environment.

2. Obtain the training image and start the container. Refer to the [Ascend image repository](https://www.hiascend.com/developer/ascendhub) to build the base image and install Rec SDK TensorFlow.
3. If you need the dynamic expansion feature in the container, see [(Optional) Installing the On-Chip Memory Dynamic Expansion Operator Package](common_operations.md#optional-installing-the-on-chip-memory-dynamic-expansion-operator-package) to build and install the dynamic expansion operator package.
4. If you need the Hadoop distributed file system, refer to the [Hadoop official documentation](https://hadoop.apache.org/docs/r1.0.4/cn/quickstart.html) for environment deployment and cluster setup. Hadoop 2.7.5 is recommended.

    >[!NOTE]
    >After you deploy the environment according to the Hadoop official documentation, the owner of the `/usr/local/hadoop-2.7.5/sbin` directory in the environment is 20415, which is a non-root user. That owner can rename and create new files to replace executables in the root user's `PATH` environment variable, which creates a privilege escalation risk.

### Building a Rec SDK TensorFlow Training Image

This section helps you build a Rec SDK TensorFlow training image from an existing base image.

**Prerequisites**

1. You have followed [Installation Prerequisites](#installation-prerequisites) to install the driver and firmware for the corresponding CANN version on the physical machine.
2. Docker has been installed on the physical machine and the Docker network is available.
3. Prepare a base image. You can obtain a base image from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub), or use an existing base image you already have.
    - You are advised to obtain the Rec SDK TensorFlow training image from the Ascend image repository. The Rec SDK TensorFlow training image in the Ascend image repository already includes GCC, CMake, and other basic dependencies, so you do not need to install them again. You only need to update the CANN and Rec SDK software packages.
    - Prepare an image as the base image. You are advised to use the Ubuntu 20.04 image.

4. Run the following command to load the base image into Docker.

    ```bash
    docker load --input xxx.tar
    ```

5. Create a folder for building the image. `build_images` is used as an example.
    1. Place only the files needed for image building in this folder, such as the `Ascend-cann-toolkit_*.run` package for the target architecture, `tf-plugin`, and the Rec SDK software package.
    2. If you need to install the `tf-plugin` package, also copy `/usr/local/Ascend/driver/version.info` and `/etc/ascend_install.info` to the `build_images` directory.

        (Do not place unrelated files in the `build_images` directory. When you build the image, files in this directory are copied into the image.)

6. The image build process uses `docker` commands and file copies from the physical machine. Make sure the user has permission to run commands and access files.

**Building a Training Image from the Rec SDK TensorFlow Base Image**

1. Refer to [Obtaining the Rec SDK TensorFlow Software Package](#obtaining-the-rec-sdk-tensorflow-software-package) to obtain the Rec SDK software package, the matching CANN software package, and the TensorFlow Ascend adaptation plugin.
2. Create a Dockerfile configuration file (`Dockerfile` for example) in the `build_images` directory. Edit the file with `vi Dockerfile` and insert the following content:

    ```bash
    # Modify the base image name and image tag as needed.
    FROM rec_sdk-tf1:7.3.0

    # CANN parameters
    ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
    ARG KERNEL_PKG=Ascend-cann-*-ops*.run
    ARG TF1_PLUGIN=npu_bridge-1.15.0-*.whl
    # Rec SDK TensorFlow package
    ARG REC_SDK_PKG=tf_rec_v2*.tar.gz

    # Set the installation path environment variable
    ARG ASCEND_BASE=/usr/local/Ascend

    # Remove the old CANN installation.
    RUN rm -rf $ASCEND_BASE/ascend-toolkit
    RUN rm -rf $ASCEND_BASE/cann*

    # Choose the dependencies to copy and install as required. If you do not need to run them, delete the corresponding lines.
    WORKDIR /tmp
    COPY $TOOLKIT_PKG .
    COPY $KERNEL_PKG .
    COPY $TF1_PLUGIN .
    COPY $REC_SDK_PKG .
    COPY version.info .
    COPY ascend_install.info .

    # Install ascend-toolkit and the ops operator package.
    RUN umask 0027 && \
        mkdir -p $ASCEND_BASE/driver && \
        /usr/bin/cp -f version.info $ASCEND_BASE/driver/ && \
        /usr/bin/cp -f ascend_install.info /etc/ && \
        chmod +x $TOOLKIT_PKG && \
        echo Y | bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
        chmod +x $KERNEL_PKG && \
        echo Y | bash $KERNEL_PKG --quiet --install && \
        source $ASCEND_BASE/cann/set_env.sh && \
        rm -rf /root/.cache/pip && \
        rm -f $TOOLKIT_PKG && \
        rm -f $KERNEL_PKG && \
        rm -rf $ASCEND_BASE/driver && \
        rm -rf /etc/ascend_install.info

    # Install Rec SDK TensorFlow.
    RUN pip3.7 install $TF1_PLUGIN --force-reinstall && \
        pip3.7 install $REC_SDK_PKG --force-reinstall && \
        rm -rf /root/.cache/pip && \
        rm -f $REC_SDK_PKG
    ```

3. Go to the `build_images` path and run the following command to build the Rec SDK TensorFlow image.

    ```bash
    docker build -t {image_name}:{image_tag} -f Dockerfile .
    ```

**Building a Training Image from Ubuntu 20.04 or a User Image**

1. Check whether the image already contains the following dependencies. Download any missing dependency package to the `build_images` directory.

    |Dependency| Download Link                                                                                        |
    |--|----------------------------------------------------------------------------------------------|
    |gcc-11.2.0| [Link](https://mirrors.ustc.edu.cn/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz)                        |
    |cmake-3.22.6| [Link](https://cmake.org/files/v3.22/cmake-3.22.6.tar.gz)                                     |
    |ucx| [Link](https://github.com/openucx/ucx/archive/master.zip)                                     |
    |openmpi-4.1.5| [Link](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz)              |
    |python-3.7.5| [Link](https://repo.huaweicloud.com/python/3.7.5/Python-3.7.5.tar.xz)                         |
    |hdf5-1.10.5| [Link](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz)|
    |CANN software package, TensorFlow Ascend adaptation plugin, and Rec SDK software package| See [Installation Prerequisites](#installation-prerequisites).                                                                                  |
    |TensorFlow (1.15.0)| [Link](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html)|

2. Create an image by referring to [OVERVIEW](../../../../docker/OVERVIEW.md).

## Configuring Environment Variables

[Table 1](#table126401659163820) describes the environment variables for Rec SDK TensorFlow. To use C/C++ compilation, such as when you compile operators written in C++, set the compilation environment variables. See [Table 2](#table20242918114315) for details.

**Table 1** Environment variables
<a id="table126401659163820"></a>

| Environment Variable                 | Meaning                         | Mandatory/Optional| Description                                                                                                                                                                                                                                                                                                                                                                                                                          |
|------------------------|-----------------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RANK_TABLE_FILE        | Communication set file used to configure Ascend chips.           | Optional   | Path of the collective communication file. Default value: `""`. This parameter is mandatory when you use the ranktable mode.                                                                                                                                                                                                                                                                                                                                                                                         |
| HCCL_TOPO_FILE_PATH    | Communication topology file for Ascend chips.           | Optional   | Path of the communication topology file. Default value: `""`. This parameter is mandatory when you use the ranktable mode.                                                                                                                                                                                                                                                                                                                                                                                         |
| ASCEND_VISIBLE_DEVICES | Devices visible to the Ascend processor, which specify that the program uses only some of them.| Mandatory   | Use the `ASCEND_VISIBLE_DEVICES` environment variable to specify the NPU devices used for training. You can run `ls /dev/ \| grep davinci*` to query the NPU devices on the host. Specify devices by serial number. Single values, ranges, and mixed forms are supported. For example:<li>`ASCEND_VISIBLE_DEVICES=0` mounts device 0 (`/dev/davinci0`) into the container. </li><li>`ASCEND_VISIBLE_DEVICES=1,3` mounts devices 1 and 3 into the container. </li><li>`ASCEND_VISIBLE_DEVICES=0-2` mounts devices 0 to 2, inclusive, into the container. This has the same effect as `ASCEND_VISIBLE_DEVICES=0,1,2`. </li><li>`ASCEND_VISIBLE_DEVICES=0-2,4` mounts devices 0 to 2 and device 4 into the container. This has the same effect as `ASCEND_VISIBLE_DEVICES=0,1,2,4`.</li> |

>[!NOTE]
>Rec SDK TensorFlow depends on the `OMPI_COMM_WORLD_SIZE`, `OMPI_COMM_WORLD_LOCAL_SIZE`, and `OMPI_COMM_WORLD_RANK` environment variables in distributed training and inference scenarios started by OpenMPI. OpenMPI launchers inject these environment variables automatically, so you do not need to inject them manually.

**Table 2** C++ build environment variables
<a name="table20242918114315"></a>

|Environment Variable|Meaning|Mandatory/Optional|Description|
|--|--|--|--|
|CC|C language compiler|Mandatory|Set it to `gcc`.|
|CXX|C++ language compiler|Mandatory|Set it to `g++`.|

## Upgrade

If you need to upgrade the current version of Rec SDK TensorFlow to the latest version, upload the latest Rec SDK software package to the installation environment, then run a command in the directory that contains the software package. See the following commands:

- When you upgrade Rec SDK TensorFlow to a new version, you must uninstall the old version first.

    ```bash
    pip3 uninstall tf_rec_v2 -y
    ```

- Use the `--upgrade` option to upgrade Rec SDK TensorFlow.

    ```bash
    pip3 install --upgrade tf_rec_v2-{version}-{arch}.tar.gz
    ```

(`{version}` indicates the version number, and `{arch}` indicates the OS architecture. Replace them with the actual ones.)

## Uninstallation

If you need to remove the Rec SDK TensorFlow software package, run the following command:

```bash
pip3 uninstall tf_rec_v2 -y
```
