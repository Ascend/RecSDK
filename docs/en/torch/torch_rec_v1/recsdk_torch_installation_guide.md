# Installation and Deployment

## Installation Instructions

### Version Mapping

Rec SDK Torch currently supports two compatible version combinations. When you install the software later, install the package that matches the corresponding combination.

| Version Combination| Python | PyTorch | torch_npu | fbgemm_gpu | Rec SDK Torch |
| -------- | ------- | ------- | --------- | ---------- | ------------- |
| Option 1  | 3.11+   | 2.6.0   | 2.6.0     | 1.1.0+cpu  | 1.1.0         |
| Option 2  | 3.11+   | 2.7.1   | 2.7.1     | 1.2.0+cpu  | 1.2.0         |

> [!NOTE]NOTE
>
> - PyTorch is a deep learning training framework. For details, see the [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).
> - torch_npu is an extension plugin that adapts the PyTorch framework to NPU devices. For details, see [torch_npu](https://gitcode.com/Ascend/pytorch).
> - fbgemm_gpu is an acceleration library that TorchRec depends on. For details, see [fbgemm_gpu](https://github.com/pytorch/FBGEMM).

### Dependency Software Description

This section describes the software dependencies and installation methods. See [Installing Rec SDK Torch](#installing-rec-sdk-torch) and install the software as needed.

**Dependency Overview**

The following table lists the software dependencies at each layer. When installing dependency software, follow the bottom-up order.

| Layer| Component| Dependencies| Description|
|------|----------|--------|------|
| **Application layer**| Rec SDK Torch | hybrid_torchrec, torchrec_embcache| Rec SDK Torch component|
| **Custom operator layer** | Packages related to custom operators| rec_ops, fbgemm_ascend| Custom operator implementation|
| **Adaptation layer**| NPU adaptation for the TorchRec framework| torchrec_npu | NPU-adapted version of TorchRec|
| **Dependency layer** | TorchRec dependency| fbgemm_gpu | Underlying acceleration library that TorchRec depends on|
| **Framework layer**| Training framework | PyTorch, torch_npu| Deep learning training framework|
| **Enablement layer**| NPU enablement| CANN | Underlying NPU support|
| **Host machine layer** | Host machine dependencies| NPU firmware, drivers, and device configuration| Basic hardware-level dependencies|

#### Host Machine Dependencies

Rec SDK Torch runs in an NPU environment. The following table describes the host machine dependency software. If the host machine does not have the required software installed, install it as described.

| Dependency/Operation              | Recommended Version             | How to Obtain/Installation Instructions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|-----------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ascend hardware product driver and firmware          | Ascend HDK 26.0.RC1 and its patch versions| Click [this link](https://www.hiascend.com/developer/download/commercial/result?module=cann), configure the package in **Edit Resource Selection** under supporting resources on the left, filter the required software package, and obtain the package after you confirm the version information.<br>For driver and firmware installation, see the [driver and firmware installation and upgrade guides](https://support.huawei.com/enterprise/en/ascend-computing/ascend-hdk-pid-252764743) that come with the hardware product.                                                                                                                                                                                                                          |
| Ascend Docker Runtime | MindCluster 7.3.0 | If Docker is not installed on the host machine, install Docker by following the [Docker community or official documentation](https://docs.docker.com/engine/install/).<br>See the **Installation** > [Installation Deployment](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_009.html) section of the *MindCluster Cluster Scheduling User Guide* to download and install the Ascend Docker Runtime package.                                                                                                                                                                                                                      |
| Configuring the device NIC           | -                 | This is host machine dependency software. Perform the operation in the host machine environment.<br>See the **Example of Parameter Network Configuration** > **Configuration Example** > [Configure Training Nodes](https://support.huawei.com/enterprise/zh/doc/EDOC1100349028/f48f446c) section of the Ascend Training Solution 23.0.0 Networking Guide 01. Use HCCN_Tool to configure Device IP address information for the NPU network port.                                                                                                                                                                                                                                                                                                         |

#### Container Training Framework Dependencies

| Dependency/Operation      | Recommended Version   | How to Obtain/Installation Instructions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CANN package         | CANN 9.0.0  | This is container dependency software. If it is not installed in the container, install it in the container.<br>Click [this link](https://www.hiascend.com/developer/download/commercial/result?module=cann), configure the package in **Edit Resource Selection** under supporting resources on the left, filter the required software package, and obtain the package after you confirm the version information.<br>Obtain `Ascend-cann-toolkit_{version}_linux-{arch}.run` and `Ascend-cann-{chip_type}-ops-{version}_linux-{arch}.run` according to the device architecture.<br><br>If you need to uninstall it, see the **Uninstallation** > **CANN Uninstallation** section in the installation guide above.|
| PyTorch and torch_npu| 2.6.0/2.7.1 | This is container dependency software. If it is not installed in the container, install it in the container.<br>According to the [Version Mapping](#version-mapping), see [Installing PyTorch and the PyTorch Ascend Adaptation Plugin](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/installation_guide/installation_via_binary_package.md) to install the PyTorch framework and the torch_npu plugin.<br>Select the installation command according to the PyTorch version, Python version, and device architecture. Python 3.11 is recommended.<br>If you need to uninstall it, run `pip3 uninstall -y torch_npu torch`.|

#### Container Training Acceleration Library Dependencies

The native TorchRec framework depends on the fbgemm_gpu library. When you run in an NPU environment, install the CPU version of fbgemm_gpu.

Because the CPU version cannot be installed through `requirements` dependencies, you must install it manually and specify the installation source.

Install the matching fbgemm_gpu package according to the [Version Mapping](#version-mapping).

| Software Version| Installation Command|
| -------- | ------------------ |
| 1.1.0+cpu  | `pip3 install fbgemm_gpu==1.1.0+cpu -i https://download.pytorch.org/whl/cpu`          |
| 1.2.0+cpu  | `pip3 install fbgemm_gpu==1.2.0+cpu -i https://download.pytorch.org/whl/cpu`          |

### Rec SDK Torch Package Description

The following table lists the Rec SDK Torch packages:

| Name                                       | Description                  |
|-------------------------------------------|----------------------|
| torch_rec_v1-*.tar.gz  | Rec SDK Torch recommendation framework package, which already includes the TorchRec Ascend registration package|
| rec_ops  | Custom operator package              |
| fbgemm_ascend                      | fbgemm custom operator package and the PyTorch framework adaptation layer   |

## Installing Rec SDK Torch

You can install the Rec SDK Torch software package in one of the following three ways:

- Option 1: Install based on containers, with host machine and container.
  - Configure the host machine, build the image, and install the Rec SDK Torch software package in the container. This option describes the full process for configuring the host environment, building the base container image, running the container, and installing the Rec SDK Torch software package.
  - **This is the recommended option.**
- Option 2: Install from source.
  - Install the Rec SDK Torch software package in the **container**. The host environment has already been configured, and you have entered the Docker container. This option describes how to install the Rec SDK Torch software package by building from source.
  - If the Docker container image you use was not built by referring to [Build the base image](../build_torch_rec_images/README.md), **incompatibilities may exist in basic software versions such as `cmake` and `glibc`**.
- Solution 3: Install using a release package.
  - Install the Rec SDK Torch software package in the **container**. The host environment has already been configured, and you have entered the Docker container. This option describes how to install the Rec SDK Torch software package and the custom operator package by downloading the release binary package.
  - If the Docker container image you use was not built by referring to [Build the base image](../build_torch_rec_images/README.md), **incompatibilities may exist in basic software versions such as `cmake` and `glibc`**.

To view the historical installation records of the Rec SDK Torch software package, see [Viewing Rec SDK Torch Installation and Uninstallation Records](../../torch/torch_rec_v1/common_operations.md).

> [!NOTE]
> For open-source and third-party software that you integrate, track vulnerabilities and issues in the community and fix them in a timely manner. You can check the [CVE website](https://www.cve.org/) for known vulnerabilities in the corresponding open-source software version, and fix them by upgrading versions or applying patch packages.

### Container-Based Installation (Host Machine + Container)

#### Brief Steps

1. Configure the host machine environment.
2. Build the base image.
3. Start the Docker container.
4. Install the Rec SDK Torch.

#### Process Flowchart

For container-based deployment of Rec SDK Torch, see [Figure 1](#fig1345216415476) for the process flowchart.

Figure 1 Development environment configuration and training image build inside the container<a id="fig1345216415476"></a>

![](../../figures/torch_rec_v1/configuring-the-development-environment-in-a-container-and-building-the-training-image.png "Configuring the development environment in a container and building the training image")

#### Detailed Steps

1. Configure the host machine environment

   See the [Host Machine Dependencies](#host-machine-dependencies) section to complete the host machine environment configuration.

2. Create the base training image.

   You can directly download the prepared base training image from the [download link](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5).

   You can also create the base training image manually. See the Dockerfile and README in [Building the Base Image](../build_torch_rec_images/README.md) to create the image.

   When you create the base image, the dependency software in [Container Training Framework Dependencies](#container-training-framework-dependencies) and [Container Training Acceleration Library Dependencies](#container-training-acceleration-library-dependencies) is installed at the same time. This includes CANN, PyTorch, torch_npu, and fbgemm_gpu. You can skip the **dependency software installation** step when you install Rec SDK Torch later.

3. Run the container.<a id="TOPIC_0000002302389300"></a>

   Create the `run_docker.sh` startup script as follows:

    ```bash
    #!/bin/bash
    container_name=$1
    image_name=$2
    docker run \
    -it \
    --name ${container_name} \
    --shm-size="300g" \
    -m 300g \
    -v /etc/localtime:/etc/localtime:ro \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    ${image_name} \
    /bin/bash
    ```

    > [!NOTE]
    > The following describes some of the parameters:
    > - `-m 300g` sets the maximum amount of memory that can be used in the container to 300 GB. Adjust it as required.
    > - `-e ASCEND_VISIBLE_DEVICES=0-7` mounts NPU devices `device0` to `device7` from the server into the container. Adjust it as required.

    Run the following command to create a container and enter it:

    ```shell
    bash run_docker.sh container_name image_name:image_version
    ```

    > [!NOTE]
    > 1. In the preceding command, the Docker container runs in the foreground. The container stops after you exit the interactive session. For more information about Docker containers, see the [Docker community documentation](https://docs.docker.com/).
    > 2. If you need to update dependency software versions in the container, see the instructions in [Container Training Framework Dependencies](#container-training-framework-dependencies) and [Container Training Acceleration Library Dependencies](#container-training-acceleration-library-dependencies). Uninstall the dependency software first and then reinstall it.

4. Install the Rec SDK Torch software package.

   To install Rec SDK Torch, refer to [Installing from Source](#installing-from-source) or [Installing Using a Release Package](#installing-using-a-release-package) below, and skip the dependency software installation step in those sections. The related dependencies are already installed when you create the base image.

### Installing from Source

> [!NOTE]
>
> The `build_wrapper.sh` script is updated with the resource download center. Currently, you are advised to install using the release package.

This option assumes that the host machine environment has already been configured and that you have entered the Docker container.

1. Install dependency software.

   See the instructions in [Container Training Framework Dependencies](#container-training-framework-dependencies) and [Container Training Acceleration Library Dependencies](#container-training-acceleration-library-dependencies) to install the dependency software in the container.

2. Install the Rec SDK Torch recommendation framework package.<a id="source_build_hybrid_torchrec"></a>

   Go to the RecSDK code directory.

   To compile and install the software package, refer to the `build/build_wrapper/torch_rec_v1/build_wrapper.sh` script and run the script to build the software package. After the build is successful, the software package is stored in the `build/output` subdirectory.

   ```bash
   # Build the package.
   bash build/build_wrapper/torch_rec_v1/build_wrapper.sh

   # Install the package.
   pip3 uninstall -y torch_rec_v1
   pip3 install build/output/torch_rec_v1*.tar.gz
   ```

   > [!NOTE]
   > The TorchRec Ascend registration package adapts the NPU device based on the TorchRec source code. When you build and install the Rec SDK Torch recommendation framework package, the TorchRec Ascend registration package is installed at the same time.
   > If you need to build and install it separately, you can use the patch file provided by Rec SDK Torch and a fixed branch of the TorchRec source code to build the registration package.
   > See the [README](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md) for source build and installation.

3. Install packages related to custom operators.

   Download the [Rec SDK](https://gitcode.com/Ascend/RecSDK) source code and build and install the operator-related packages with the following commands:

   ```bash
   # Before building the operators, enable the CANN environment variables. When you install the CANN package in the default path, run the following command to enable the CANN environment variables:
   source /usr/local/Ascend/cann/set_env.sh
   unset ASCEND_CUSTOM_OPP_PATH

   # Build and install the operator package (rec_ops). The A2 operator is used as an example.
   cd RecSDK
   git submodule update --init --recursive
   cd cust_op/ascendc_op/build
   bash build_ai_core_op.sh A2

   # Optional: If you only need to install some operators, you can build them in another container, copy the required operator package from build/output/recsdk_ops into the current environment, and install it by running the following command:
   # bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
   ```

   The parameters for installing the operator `run` package are shown in [Table 1](#table14435173717221).

   **Table 1**  Parameters
   <a id="table14435173717221"></a>

   |Parameter|Description|
   |--|--|
   |--help \| -h|Queries help information.|
   |--info|Queries package information.|
   |--list|Queries the file list in the package.|
   |--check|Checks package integrity.|
   |--quiet|Uses the silent installation mode.|
   |--nox11|Does not start the xterm terminal.|
   |--noexec|Does not run the embedded installation script.|
   |--extract=\<path>|Directly extracts the package to the target directory. This is usually used together with `--noexec` to extract files only and not run the script.|
   |--tar arg1 [arg2 ...]|Accesses the package contents through the `tar` command.|
   |--install-path|Installs to the specified directory path.|

   >[!NOTICE]
   >
   > After you install the operators, the `/usr/local/Ascend/cann/opp/vendors/` directory contains folders such as `split_embedding_codegen_forward_unweighted`, `backward_codegen_adagrad_unweighted_exact`, `asynchronous_complete_cumsum`, and `permute2d_sparse_data`. If the corresponding folders do not exist, run `unset ASCEND_CUSTOM_OPP_PATH` to clear the environment variable and then install the operators again.

4. Install the `fbgemm_ascend` operator and its adaptation layer

   See the [README](https://gitcode.com/Ascend/fbgemm-ascend/blob/main/README.md) for `fbgemm_ascend` to build and install it from source.

### Installing Using a Release Package

This option assumes that the host machine environment has already been configured and that you have entered the Docker container.

The installation steps for this option are similar to those in [Installing from Source](#installing-from-source). The difference is that you can obtain the Rec SDK Torch recommendation framework package directly from the release package, without building it from source.

1. Install dependency software.

   See the instructions in [Container Training Framework Dependencies](#container-training-framework-dependencies) and [Container Training Acceleration Library Dependencies](#container-training-acceleration-library-dependencies) to install the dependency software in the container.

2. Install the Rec SDK Torch recommendation framework package.

   **Downloading the Software Package**

   Refer to this section to obtain the required software package and the corresponding digital signature file. By downloading the software, you agree to the terms and conditions of [Huawei Enterprise End User License Agreement (EULA)](https://e.huawei.com/en/about/eula).

   > [!NOTE]
   > The current release software package is the Rec SDK .whl package. The one-click installation and deployment software package will be updated after the resource download center is launched.

   | Component                | Software Package                                     | Link                                              |
   |----------------------|------------------------------------------|----------------------------------------------------|
   | Rec SDK Torch recommendation framework package| torch_rec_v1-*.tar.gz | [Link](https://gitcode.com/Ascend/RecSDK/releases)|

   > [!NOTE]
   > The current Rec SDK recommendation framework package is built with Python 3.11. **Install and use it in the same Python version environment.** If you need to use it in another Python version environment, see [Install the Rec SDK Torch recommendation framework package](#source_build_hybrid_torchrec) and build it from source.

   **Software package hash verification**

   To prevent malicious tampering during transmission or storage, use the `sha256sum` command after you download the software package to verify that the hash value matches the one on the package download page.

   **Install the software package**

   Upload the downloaded package into the Docker container. You can do this in either of the following ways:

   - Method 1: When you start the container, mount a host machine directory into the container and place the downloaded package in that directory so the container can access it.

       To mount the host machine directory into the Docker container, add the following parameter when you [run the container](#TOPIC_0000002302389300). Replace `dirX` with the actual directory:

       ```bash
       -v /dir1:/dir1
       ```

   - Method 2: On the host machine, use the `docker cp` command to copy the package into the container.

       Example of the `docker cp` command:

       ```bash
       docker cp host_file_path container_name:container_file_path
       ```

       Here, `host_file_path` is the file path on the host machine. `container_name` is the name of the Docker container into which you want to copy the file. `container_file_path` is the file path inside the Docker container.

   Run the following command to install:

   ```shell
   # If it is already installed, uninstall it first.
   pip3 uninstall -y torch_rec_v1
   # Install the software package.
   pip3 install torch_rec_v1-{version}-{arch}.tar.gz
   ```

   (`{version}` indicates the version number, and `{arch}` indicates the OS architecture. Replace them with the actual ones.)

3. Install packages related to custom operators.

   See the [README](https://gitcode.com/Ascend/fbgemm-ascend/blob/main/README.md) for `fbgemm_ascend` to obtain the `fbgemm_ascend` software package.

   > [!NOTE]
   > The `fbgemm_ascend` wheel package and the `rec_ops` wheel package will be updated after the resource download center goes live.

   ```shell
   # # Install the framework dependency operator package fbgemm_ascend.
   pip3 install fbgemm_ascend-*.whl
   # # Install the custom operator package rec_ops.
   pip3 install rec_ops-*.whl
   ```

## Verifying the Installation

You can verify that Rec SDK Torch installed successfully by running existing test cases.

For the `hybrid_torchrec` test case list and instructions for running the tests, see [README](../../../../training/torch_rec_v1/hybrid_torchrec/test/st/README.md).

For the `torchrec_embcache` test case list and instructions for running the tests, see [README](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/README.md).

After you run the test cases, if `xx passed` is displayed and `xx failed` is not displayed, the test cases passed and Rec SDK Torch installed successfully.

## Configuring Environment Variables

[Table 1](#table126401659163820) describes the environment variables for Rec SDK Torch.

**Table 1** Environment variables
<a id="table126401659163820"></a>

| Environment Variable                          | Meaning                                                 | Mandatory/Optional| Description                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------------------------------|-----------------------------------------------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| INPUT_DIST_THREADS              | Number of concurrent threads in the thread pool that Rec SDK Torch uses for bucketization tasks.                       | Optional   | The value is an integer. The default value is 6. The value range is [1, 12].                                                                                                                                                                                                                                                                                                                                                                                                                    |
| POST_INPUT_THREADS              | Number of concurrent threads in the thread pool that Rec SDK Torch uses for hash deduplication tasks.                     | Optional   | The value is an integer. The default value is 6. The value range is [1, 12].                                                                                                                                                                                                                                                                                                                                                                                                                    |
| MASTER_ADDR                     | IP address of the master node in distributed training.                                | Optional   | The value is an IPv4 address. 127.0.0.1 is recommended.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| MASTER_PORT                     | Listening port in distributed training.                                   | Optional   | The value is an integer. The value range is [0, 65520].                                                                                                                                                                                                                                                                                                                                                                                                                       |
| LOCAL_RANK                      | NPU rank of the current process on the local machine.                                    | Optional   | The value is an integer. The value range is [0, world_size-1].                                                                                                                                                                                                                                                                                                                                                                                                               |
| WORLD_SIZE                      | Number of devices that participate in training.                                     | Optional   | The value is an integer. The value range is [1, 8].                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ASCEND_VISIBLE_DEVICES          | Devices visible to the Ascend processor, which specify that the program uses only some of them.                        | Mandatory   | Use the `ASCEND_VISIBLE_DEVICES` environment variable to specify the NPU devices used for training. You can run `ls /dev/ \| grep davinci *` to query the NPU devices on the host. Specify devices by serial number. Single values, ranges, and mixed forms are supported. For example:<ul><li>`ASCEND_VISIBLE_DEVICES=0` mounts device 0 (`/dev/davinci0`) into the container. </li><li>`ASCEND_VISIBLE_DEVICES=1,3` mounts devices 1 and 3 into the container. </li><li>`ASCEND_VISIBLE_DEVICES=0-2` mounts devices 0 to 2, inclusive, into the container. This has the same effect as `ASCEND_VISIBLE_DEVICES=0,1,2`. </li><li>`ASCEND_VISIBLE_DEVICES=0-2,4` mounts devices 0 to 2 and device 4 into the container. This has the same effect as `ASCEND_VISIBLE_DEVICES=0,1,2,4`.</li></ul> |
| ASCEND_OPP_PATH                 | Root directory of the operator library.                                            | Mandatory   | This variable is set when you run the CANN environment variable configuration script. You are advised not to modify it. The default value is `/usr/local/Ascend/cann/opp`.                                                                                                                                                                                                                                                                                                                                                                                 |
| GLOO_SOCKET_IFNAME              | gloo communication NIC configuration.                                        | Optional   | Run the `ifconfig` or `ip a` command to view the NIC name on the server. `lo` is recommended.                                                                                                                                                                                                                                                                                                                                                                                              |
| ENABLE_FAST_HASHMAP             | Indicates whether to enable the fast hash table.                                          | Optional   | The value is a string. `true`, `yes`, and `1` enable the function. Other values disable the function. The default value is `false`.                                                                                                                                                                                                                                                                                                                                                                                            |
| EMB_MEMORY_POOL_SIZE            | Size of the embedding memory pool for the fast hash table.                               | Optional   | The value is an integer. The default value is 102400. The value range is [1, 200000].                                                                                                                                                                                                                                                                                                                                                                                                           |
| FAST_HASHMAP_RESERVE_BUCKET_NUM | Number of reserved buckets in the fast hash table.                                         | Optional   | The value is an integer. The default value is 2097152. The value range is [128, 4294967291].                                                                                                                                                                                                                                                                                                                                                                                                    |
| EMB_MEMORY_POOL_THREAD_NUM      | Number of threads that process the embedding memory pool of the fast hash table.                             | Optional   | The value is an integer. The default value is 4. The value range is [1, 1024].                                                                                                                                                                                                                                                                                                                                                                                                                  |
| EMBCACHE_SIZE_ON_DEVICE_MEM     | On-chip memory embedding cache size, in bytes.                           | Optional   | The value is an integer. The default value is 17179869184 (16 GB). The value range is [1, available device memory].                                                                                                                                                                                                                                                                                                                                                                                                |
| DO_EC_LOCAL_UNIQUE              | Indicates whether to enable EC local unique for the multi-level cache mode.                            | Optional   | The value is a string. `true`, `1`, and `yes` enable the function. Other values disable the function. The default value is `false`.                                                                                                                                                                                                                                                                                                                                                                                            |
| LOCAL_UNIQUE_PARALLEL_BATCH_NUM | Number of batches processed in parallel by local unique in `EmbCacheTrainPipelineSparseDist`.| Optional   | The value is an integer. The default value is 2. The value range is [1, 24].                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ENABLE_PARALLEL_GLOBAL_UNIQUE   | Indicates whether to enable parallel global unique processing.                              | Optional   | The value is a string. `1` enables the function. Other values disable the function. The default value is `0`, indicating that the function is disabled.                                                                                                                                                                                                                                                                                                                                                                                                           |
| GLOG_stderrthreshold            | Log level of the multi-level cache C++ module.                                  | Optional   | The value is an integer. The default value is `1` (WARN). Value range:<ul><li>`-2`: TRACE</li><li>`-1`: DEBUG</li><li>`0`: INFO</li><li>`1`: WARN</li><li>`2`: ERROR</li></ul>                                                                                                                                                                                                                                                                                                                                    |

## Uninstallation

If you need to remove the Rec SDK Torch recommendation framework package and the TorchRec Ascend registration package, run the following command:

```bash
# Uninstall the Rec SDK Torch main package and the TorchRec Ascend registration package together.
pip3 uninstall torch_rec_v1 -y
```

If you need to remove the Rec SDK Torch custom operator-related packages, run the following command. In this case:

- The default installation path for custom operators in CANN is `/usr/local/Ascend/cann/opp/vendors/`
- When you uninstall custom operators, delete the folder in the `vendors` path whose name matches the custom operator. You can view the Rec SDK custom operator directory in [ai_core_op](https://gitcode.com/Ascend/RecSDK/tree/develop/cust_op/ascendc_op/ai_core_op).

```bash
# Uninstall for source build installation.
# Example of the command to uninstall an operator. Only some examples are listed here. The uninstall commands for other operators are similar.
rm -rf /usr/local/Ascend/cann/opp/vendors/asynchronous_complete_cumsum
rm -rf /usr/local/Ascend/cann/opp/vendors/backward_codegen_adagrad_unweighted_exact
rm -rf /usr/local/Ascend/cann/opp/vendors/permute2d_sparse_data
rm -rf /usr/local/Ascend/cann/opp/vendors/split_embedding_codegen_forward_unweighted
# Uninstall libfbgemm_npu_api.so.
PACKAGE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [ -d "$PACKAGE_PATH" ]; then
  cd ${PACKAGE_PATH}
  rm -rf libfbgemm_npu_api.so
else
  echo "no site-package"
fi

# Uninstall for installation using a release package.
pip uninstall rec_ops
pip uninstall fbgemm_ascend
```

## Upgrading

If you need to upgrade the current version of Rec SDK Torch to the latest version, upload the latest Rec SDK Torch software package to the installation environment, then run a command in the directory that contains the software package. See the following commands:

1. When you upgrade Rec SDK Torch to a new version, you must uninstall the old version first. For details, see [Uninstallation](#uninstallation).
2. Reinstall Rec SDK Torch by referring to [Installing Rec SDK Torch](#installing-rec-sdk-torch).
