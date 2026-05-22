# torchrec_npu Docker 镜像

基于 Ubuntu 22.04，预装 CANN 9.0.0 + PyTorch + torch_npu + torchrec 的 NPU 开发镜像。

## 软件栈

| 虚拟环境 | PyTorch | torch_npu | torchrec | fbgemm_gpu |
|----------|---------|-----------|----------|------------|
| torchrec1.2.0 | 2.7.1 | 2.7.1 | 1.2.0 | 1.2.0 |
| torchrec1.5.0 | 2.10.0 | 2.10.0 | 1.5.0 | 1.5.0 |

## 前期准备

构建前需要根据系统架构（x86_64或aarch64）在 Dockerfile 同级目录下准备好以下文件：

```tree
docker/torchrec_npu/
├── Dockerfile
├── fbgemm_ascend-1.2.0*.whl
├── fbgemm_ascend-1.5.0*.whl
├── torch-npu-2.7.1*.whl
└── torch-npu-2.10.0*.whl
```

## 构建镜像

```bash
cd docker/torchrec_npu
docker build -t torchrec_npu:latest .
```

## 启动容器

```bash
docker run -it \
    --name {容器名} \
    --net=host \
    -m 300g \
    -e ASCEND_VISIBLE_DEVICES=0-7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v {挂载目录}:{挂载目录} \
    {镜像名}:{镜像tag} \
    /bin/bash
```

参数说明：

- ASCEND_VISIBLE_DEVICES=0-7 当前机器所拥有的npu卡为0-7卡，如果是16卡可以设置为0-15，用户可根据实际情况挂载
- -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ 容器挂载的驱动目录，用户可按照实际驱动的安装目录进行配置
- -v /etc/ascend_install.info:/etc/ascend_install.info 容器需挂载驱动固件安装信息
- -m 300g 设置容器内可用内存，可根据使用情况进行配置

## 激活虚拟环境

进入容器后，激活CANN包环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

根据需要激活对应的 torchrec 环境：

```bash
# torchrec 1.2.0
source /opt/buildtools/torchrec1.2.0/bin/activate

# torchrec 1.5.0
source /opt/buildtools/torchrec1.5.0/bin/activate
```
