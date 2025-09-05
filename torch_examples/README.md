# RecSDK-Torch 模型样例运行环境说明

## 版本配套说明
本模型迁移依赖特定版本的CANN、PyTorch、驱动和固件,源码编译需使用指定版本的Python、GCC、CMake等工具,仅支持昇腾平台（Atlas 800T A2）,基于软件环境以RecSDK-Torch提供的基础镜像环境为准，主要的配套依赖如下表所示：

| Python版本   | 主要配套依赖                                                                                                         |
|------------|----------------------------------------------------------------------------------------------------------------|
| Python3.11 | torch==2.6.0<br/>torch_npu==2.6.0<br/>fbgemm_gpu==1.1.0+cpu<br/>torchrec==1.1.0+npu<br/>hybrid_torchrec==1.1.0 |

## 基础镜像
下载基础镜像地址为：https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5

## 启动容器
说明：以下启动命令仅作参考，按需挂载目录。
```shell
#!/bin/bash
container_name=$1
image_name=$2
docker run \
-it \
--name "${container_name}" \
-e ASCEND_VISIBLE_DEVICES=0-7 \
--shm-size="300g" \
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```
执行如下命令新建容器：
```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

## 设置环境变量
进入容器后，设置环境变量
```shell
source /etc/profile
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装依赖
说明：容器中已经安装好torchrec,hybrid_torchrec以及算子等依赖。如需重新安装依赖需确保网络通畅，可选择（1）通过获取安装包或者（2）源码编译的方式。

1.获取安装包安装

获取地址：https://gitcode.com/Ascend/RecSDK/releases
```shell
# 如果已经安装,请先卸载
pip3 uninstall -y hybrid_torchrec torchrec
# 安装torchrec
tar -zxvf Ascend-mindxsdk-torchrec-1.1.0-npu-*.tar.gz
pip3 install torchrec-1.1.0+npu-*.whl
pip3 install -r requirements.txt

# 安装hybrid_torchrec
tar -zxvf Ascend-mindxsdk-hybrid-torchrec-1.1.0-*.tar.gz
pip3 install hybrid_torchrec-1.1.0-*.whl

# 安装所需算子
tar -zxvf Ascend-recsdk-npu-ops-*.tar.gz
cd recsdk-npu-ops/recsdk_ops/
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_gather_for_rank1.run
bash mxrec_opp_hstu_dense_forward.run
bash mxrec_opp_hstu_dense_backward.run

# 编译算子适配文件
cd ../../
cd recsdk-npu-ops/torch_plugin/torch_library/2.6.0/common
bash build_ops.sh
```
2.源码编译安装

（1）编译安装torchrec
参考：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_npu/README.md

（2）编译安装hybrid_torchrec
参考：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/hybrid_torchrec/README.MD

（3）编译安装算子和适配文件
参考：https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/build/README.md

编译完成后生成的tar包在 RecSDK/cust_op/ascendc_op/output下,安装算子和算子适配文件请参考以上安装方法。


注意：执行完"编译算子适配文件"步骤后，融合算子的依赖包libfbgemm_npu_api.so会生成在同目录下的build文件夹下，同时也会生成在python默认安装的site-package路径中，也可以将该so包拷贝到指定的目录下，在后续模型运行时会配置该文件的路径 。
