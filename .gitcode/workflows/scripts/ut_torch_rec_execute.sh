#!/bin/bash
set -e
serviceName=$1

export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.11.0/bin:$PATH
echo "nameserver 8.8.8.8" >> /etc/resolv.conf

cd ${ATOMGIT_WORKSPACE}
echo "安装rec_cust_ops和fbgemm"
REC_CUST_OBS_URL="https://mindcluster.obs.cn-north-4.myhuaweicloud.com/newRecSDK/daily/version/26.2.0/20260923.9/rec_cust_ops-2.7.1-cp311-cp311-linux_x86_64.whl"
FBGEMM_OBS_URL="https://mindcluster.obs.cn-north-4.myhuaweicloud.com/fbgemm-ascend/daily/version/26.2.0/20260928.2/fbgemm_ascend-1.2.0-cp311-cp311-linux_x86_64.whl"

# 临时目录
TMP_WHL_DIR="./tmp_whl"
mkdir -p ${TMP_WHL_DIR}

REC_WHL="${TMP_WHL_DIR}/rec_cust_ops-2.7.1-cp311-cp311-linux_x86_64.whl"
FBGEMM_WHL="${TMP_WHL_DIR}/fbgemm_ascend-1.2.0-cp311-cp311-linux_x86_64.whl"

echo "===== 开始下载 rec_cust_ops whl ====="
wget --no-check-certificate -O "${REC_WHL}" "${REC_CUST_OBS_URL}"

echo "===== 开始下载 fbgemm_ascend whl ====="
wget --no-check-certificate -O "${FBGEMM_WHL}" "${FBGEMM_OBS_URL}"

echo "===== pip安装两个包 ====="
pip install "${REC_WHL}" "${FBGEMM_WHL}"

# 清理临时文件
rm -rf ${TMP_WHL_DIR}
echo "===== 安装全部完成 ====="

cd ${ATOMGIT_WORKSPACE}/${serviceName}/training/torch_rec_v1/torchrec_npu
ls -al ./
dos2unix *.sh && chmod +x *
cp ${ATOMGIT_WORKSPACE}/opensource/torchrec-release-v1.2.0.zip ${ATOMGIT_WORKSPACE}/${serviceName}/training/torch_rec_v1/torchrec_npu/
# curl --version
unzip torchrec-release-v1.2.0.zip
mv torchrec-release-v1.2.0 torchrec
ls -al torchrec
bash build_whl_torchrec1.2.0.sh
cd torchrec/dist/ && pip3 install torchrec-1.2.0+npu-py3-none-linux_x86_64.whl --force-reinstall --ignore-requires-python --no-deps
gcc --version
export PATH=/usr/local/bin/cmake:$PATH
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cmake --version
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.11.0/bin:$PATH
. /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ${ATOMGIT_WORKSPACE}/${serviceName}/training/torch_rec_v1/hybrid_torchrec && ls -al && dos2unix *.sh && chmod +x *
source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
bash build_whl.sh
deactivate
tar -xzf Ascend-mindxsdk-hybrid-torchrec-1.2.0-pytorch2.7.1-linux-x86_64.tar.gz -C ./
ls -al
echo '开始安装前置依赖'
pip3 install fbgemm-gpu-cpu==1.2.0
echo '完成安装fbgemm'
pip3 install pytest pytest-cov pyyaml pytz tensordict
pip3 install hybrid_torchrec-1.2.0-py3-none-linux_x86_64.whl --force-reinstall --ignore-requires-python
cd ${ATOMGIT_WORKSPACE}/${serviceName}/training/torch_rec_v1/hybrid_torchrec/test/dt/ && dos2unix *.sh && chmod +x *
pip3 install --trusted-host mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple parameterized
pip3 install lxml -i https://repo.huaweicloud.com/repository/pypi/simple
pip3 install bs4 -i https://repo.huaweicloud.com/repository/pypi/simple
pip3 install urllib3==1.26.5 -i https://repo.huaweicloud.com/repository/pypi/simple
bash test_all.sh
ls -la ${ATOMGIT_WORKSPACE}/${serviceName}/training/torch_rec_v1/hybrid_torchrec/test/dt/
cd ${ATOMGIT_WORKSPACE}/ci/mindxsdk/script/
ls -la ./
