# WuKong模型NPU适配

## 适配说明

本样例的适配对象为Wukong Recommendation模型, 将其迁移至NPU侧进行推理;

模型参考的开源链接为 https://github.com/clabrugere/wukong-recommendation

克隆源码并固定版本为:Commits on Apr 30, 2024，提交的SHA-1 hash值（提交ID）：4a2cbc8adbdb4565d854a9977ee7b53175c0bc7c

验证运行的算力平台：Atlas A2推理系列产品

## 版本配套说明

### 软件配套说明

| 软件包简称   | 配套版本   |
|---------|--------|
| Python  | 3.11.0 |
| Pytorch | 2.6.0  |


### CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
|--------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| CANN-toolkit | 8.2.RC1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.2.RC1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.2.RC1&driver=Ascend+HDK+25.2.0

## 安装依赖

请根据机器架构、机器型号选择合适的安装包进行安装。

### 安装Pytorch配套

```shell
# torch版本
pip3 install torch==2.6.0+cpu  --index-url https://download.pytorch.org/whl/cpu # x86
pip3 install torch==2.6.0 # arm
# torch_npu
pip3 install torch_npu-2.6.0.*.whl
```

## 模型运行
需要把run.py文件拷贝到wukong工程文件的根目录

```
cp run.py wukong-recommendation/
cd wukong-recommendation/
python3 run.py
```

### 性能参考
run.py脚本会在当前目录下的result脚本中生成profiling文件；

## FAQ
