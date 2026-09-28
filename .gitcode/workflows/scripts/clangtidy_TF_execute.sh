#!/bin/bash
set -e
serviceName=$1


# 读取变更文件列表
MODIFY_FILE="change.txt"
FILE_ARRAY=()
exec 3< "${MODIFY_FILE}"
while IFS= read -r raw_line <&3 || [[ -n "${raw_line}" ]]; do
    fp=$(echo "${raw_line}" | sed -e 's/^"//' -e 's/"$//' -e 's/^[ \t]*//' -e 's/[ \t]*$//')
    if [[ -n "${fp}" ]]; then
        FILE_ARRAY+=("${fp}")
    fi
done
exec 3<&-

# 过滤只保留源码文件传入检测
CHECK_ARRAY=()
for file in "${FILE_ARRAY[@]}"; do
    if [[ "${file}" == *training/torch_rec_v2* ]] && [[ "${file}" =~ [.](c|h|cpp|hpp|cc|hh|cxx|hxx)$ ]]; then
        CHECK_ARRAY+=("${file}")
    fi
done

if [[ ${#CHECK_ARRAY[@]} -eq 0 ]]; then
    echo "INFO: 本次变更没有 .cpp / .cc / .cxx / .c 文件，无需代码检查，脚本退出"
    exit 0
fi

echo "====== 开始准备pybind与securec"
cd ${ATOMGIT_WORKSPACE}/opensource
unzip -q pybind11-2.10.3.zip
mv pybind11-2.10.3 pybind11

unzip -q huaweicloud-sdk-c-obs-3.23.9.zip
mv huaweicloud-sdk-c-obs-3.23.9/platform/huaweisecurec securec
rm -rf huaweicloud-sdk-c-obs-3.23.9
rm -rf securec/lib/*
cd ${ATOMGIT_WORKSPACE}/opensource/securec/src && make -j8 -s

cd ${ATOMGIT_WORKSPACE}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
. /etc/profile
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib:/usr/local/Ascend/driver/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
echo "====== 开始下载tf_plugin"
wget --no-check-certificate https://mindcluster.obs.cn-north-4.myhuaweicloud.com/blueImageDependency/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl
pip3 install npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl --no-deps

echo "==== 开始执行clangtidy脚本"
cd ${ATOMGIT_WORKSPACE}/${servicename}/build
ls -la ./
chmod +x clang_tidy_check.sh
dos2unix clang_tidy_check.sh
bash clang_tidy_check.sh tf_v1
