
# RecSDK构建镜像（使用Dockerfile）
## 前提
- 物理机上已经安装好对应CANN版本的驱动和固件
- 物理机上已经安装docker，并且docker网络可用
- 准备基础OS镜像：在物理机上使用命令`docker pull debian:12`从Dockerhub上拉取debian12镜像

### 构建步骤
Step1: 新建`build_images`目录。

Step2：在`build_images`目录下准备CANN包。用户可以从[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26)下载**8.1.RC1.beta1**版本的[toolkit](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1.beta1)包与[kernels](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1.beta1)包。用户也可根据实际情况选择其他CANN包。

安装CANN包需要两个文件，分别是
- version.info（驱动版本文件）
- ascend_install.info（固件驱动安装文件）

可将物理机上相应的文件拷贝到`build_images`目录。物理机安装驱动与固件后，version.info默认安装路径为/usr/local/Ascend/driver/version.info；
ascend_install.info默认安装路径为/etc/ascend_install.info。

Step3：将Dockerfile移动到`build_images`目录中，并运行下面命令构建镜像。构建镜像的步骤在Dockerfile中有详细的说明，注释部分是安装CANN包与torchrec相关包的操作。
```dockerfile
# 服务器能访问外网
docker build -t recsdk_torch_base:v1.0-x86 -f Dockerfile .
# 服务器配置代理访问外网
docker build -t recsdk_torch_base:v1.0-x86 -f Dockerfile --build-arg http_proxy=http://your_proxy --build-arg https_proxy=https://your_proxy .
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
-u root \
-it \
--name ${container_name} \
--shm-size="300g" \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
```
执行如下命令新建容器：
```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

进入容器后，执行：
```shell
sudo su
```
切换为root用户
```

## 安装RecSDK相关的包
参考RecSDK/README_TORCH.md进行源码的编译和安装