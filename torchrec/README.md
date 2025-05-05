# torchrec适配npu说明

## 代码结构
```shell
.
├── build_whl.sh            # torchrec==1.1.0+npu构建脚本
├── torchrec_npu.patch      # 侵入式修改patch文件
└── README.md               # 适配说明
```
## 编译安装
```shell
# 编译
bash build_whl.sh
# 安装
cd dist
pip3 install torchrec-1.1.0+npu-*.whl
pip3 install -r requirements.txt
```
## 说明
1.本样例提供了一个简洁的torchrec适配到昇腾设备方案，用户可快速编译、安装torchrec的npu适配版本。

2.本样例是基于开源torchrec项目v1.1.0版本基础上进行的适配。配套使用的torch版本为2.6.0,fbgemm_gpu版本为1.1.0。
