
# RecSDK构建镜像（使用Dockerfile）
## 前提
- 物理机上已经安装好对应CANN版本的驱动和固件
- 物理机上已经安装docker，并且docker网络可用
- 准备基础OS镜像：在物理机上使用命令
  - debian: x86架构: `docker pull debian:12` 从Dockerhub上拉取镜像
  - openeuler: arm架构: `wget https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/docker_img/aarch64/openEuler-docker.aarch64.tar.xz && docker load -i openEuler-docker.aarch64.tar.xz`<br>
  - centos: x86架构: `docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/centos:7.6.1810`<br>

## 版本配套说明

当前Rec SDK Torch支持两种版本配套：

| 配套版本  | PyTorch | torch-npu | fbgemm_gpu |
|-------|---------|-----------|------------|
| 配套版本1 | 2.6.0   | 2.6.0     | 1.1.0      |
| 配套版本2 | 2.7.1   | 2.7.1     | 1.2.0      |

在dockerfile文件中，会下载PyTorch软件包和fbgemm_gpu软件包。（torchrec-npu软件包为源码编译，不在dockerfile安装范围内）

当前dockerfile中默认配置下载PyTorch 2.6.0和fbgemm_gpu 1.1.0+cpu版本的软件包。

若需制作PyTorch 2.7.1版本配套的镜像，可参考后续[制作PyTorch 2.7.1镜像环境](#制作pytorch-271镜像环境)章节。

## 构建步骤
Step1：新建`build_images`目录。

Step2：在`build_images`目录下准备CANN包。用户可以从[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26)下载**8.2.RC1**版本的[toolkit](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)包与[kernels](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)包。用户也可根据实际情况选择其他CANN包。

安装CANN包需要两个文件，分别是
- version.info（驱动版本文件）
- ascend_install.info（固件驱动安装文件）

可将物理机上相应的文件拷贝到`build_images`目录。物理机安装驱动与固件后，version.info默认安装路径为/usr/local/Ascend/driver/version.info；
ascend_install.info默认安装路径为/etc/ascend_install.info。

Step3：将Dockerfile移动到`build_images`目录中，并运行下面命令构建镜像。构建镜像的步骤在Dockerfile中有详细的说明，注释部分是安装CANN包与torchrec相关包的操作。
```dockerfile
# 服务器能访问外网
docker build -t recsdk_torch_base:v1.0-[x86|arm] -f Dockerfile_[centos|debian|openeuler] .
# 服务器配置代理访问外网
docker build -t recsdk_torch_base:v1.0-[x86|arm] -f Dockerfile_[centos|debian|openeuler] --build-arg http_proxy=http://your_proxy --build-arg https_proxy=https://your_proxy .
```
**注意：**   
1. 制作镜像时需确保服务器能访问外网，否则需要配置代理。
2. 运行命令前注意修改Dockerfile中的基础镜像名。

## 启动容器
创建启动脚本run_docker.sh，参考如下：
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
部分参数说明：
- -m 300g：设置容器内使用内存大小，可根据实际情况进行配置。
- -e ASCEND_VISIBLE_DEVICES=0-7：将服务器上编号为device0-device7的NPU设备挂载到容器内，可根据实际情况进行配置。

执行如下命令新建容器：
```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

## 安装RecSDK相关的包
参考RecSDK/README_TORCH.md进行源码的编译和安装

## 制作PyTorch 2.7.1镜像环境
可参考以下两种方式制作PyTorch 2.7.1版本的镜像环境。

### 方式一：修改dockerfile后重新制作镜像
基于当前已有的dockerfile, 修改部分内容后编译镜像。
1. 修改PyTorch、fbgemm_gpu软件的下载链接。下载链接如下：

| 软件版本                   | 基于Python版本 | 下载链接                                                                                               |
|------------------------|------------|----------------------------------------------------------------------------------------------------|
| PyTorch 2.7.1 (X86)    | 3.11       | https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl       |
| PyTorch 2.7.1 (ARM)    | 3.11       | https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl      |
| fbgemm_gpu 1.2.0 (X86) | 3.11       | https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.2.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl  |
| fbgemm_gpu 1.2.0 (ARM) | 3.11       | https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.2.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl |

2. 修改安装PyTorch、fbgemm_gpu、torch-npu包安装时的包名

| 修改前                        | 修改后                        |
|----------------------------|----------------------------|
| torch-2.6.0+cpu-*.whl      | torch-2.7.1+cpu-*.whl      |
| fbgemm_gpu-1.1.0+cpu-*.whl | fbgemm_gpu-1.2.0+cpu-*.whl |
| torch-npu==2.6.0           | torch-npu==2.7.1           |

3. 参考[构建步骤](#构建步骤)章节制作镜像

### 方式二：升级已有镜像中的软件版本
1. 升级PyTorch版本
```shell
pip3 install torch==2.7.1+cpu -i https://download.pytorch.org/whl/cpu
```
2. 升级fbgemm_gpu版本
```shell
pip3 install fbgemm_gpu==1.2.0+cpu -i https://download.pytorch.org/whl/cpu
```
3. 升级torch-npu版本
```shell
pip3 install torch-npu==2.7.1
```
