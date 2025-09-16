## FuxiCTR模型库具有以下依赖项：
GitHub地址：https://github.com/reczoo/FuxiCTR/

Python 3.9+
pytorch 1.10.0--2.1.2（用于 torch 模型）
请通过 安装其他必需的软件包。pip install -r requirements.txt

可能出现sklearn版本过高的情况 可通过降版本至 sklearn<1.5 解决
详情见 https://github.com/reczoo/FuxiCTR/issue/156

## 运行环境准备
请参考：https://gitcode.com/Ascend/RecSDK/tree/develop/torch_examples/README.md

## fuxictr模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。
```shell
git clone -b main https://github.com/reczoo/FuxiCTR.git
cd FuxiCTR && git checkout b7dff73
cp -f ../fuxictr_npu.patch ./ && git apply fuxictr_npu.patch
```