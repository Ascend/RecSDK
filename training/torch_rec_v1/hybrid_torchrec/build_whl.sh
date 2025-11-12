#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e
SCRIPT_PATH=$(cd $(dirname $0); pwd)
TORCHREC_EMBCACHE_PATH="${SCRIPT_PATH}/../torchrec_embcache"

function get_pytorch_ver()
{
    local v260="2.6.0"
    local v271="2.7.1"
    local ver
    ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null) || {
        echo "ERROR: failed to get torch version - $ver" >&2
        return 1
    }
    case "$ver" in
        *"$v260"*) echo "pytorch${v260}" ;;
        *"$v271"*) echo "pytorch${v271}" ;;
    esac
}

function get_torchrec_ver()
{
    local v110="1.1.0"
    local v120="1.2.0"
    local ver
    ver=$(python3 -c "import torchrec; print(torchrec.__version__)" 2>&1)
    case $? in
        0) ;;          # 导入成功，继续判断版本
        *) echo "$v110"; return 0 ;;  # 模块不存在，直接返回 1.1.0
    esac
    case "$ver" in
        *"$v110"*) echo "$v110" ;;
        *"$v120"*) echo "$v120" ;;
        *) echo "ERROR: torchrec version is not supported, only support $v110 or $v120." >&2; return 1 ;;
    esac
}

function build_torchrec_embcache()
{
    cd ${TORCHREC_EMBCACHE_PATH}
    bash build.sh
    cd -
    cp ${TORCHREC_EMBCACHE_PATH}/dist/torchrec_embcache-*.whl dist/
}

function build_hybrid_torchrec()
{
    local cur_path=${PWD}
    cd ${PWD}/src/
    bash run.sh
    cp ./build/*.so ${cur_path}/hybrid_torchrec/modules/
    cd -
    if [ -d "dist" ]; then
        rm -rf ./dist
    fi
    python3 setup.py bdist_wheel --plat-name linux_"${ARCH}"
    cp requirements.txt dist/
}

function archive_target_pkg()
{
    local pt_version=$(get_pytorch_ver)
    local package_name="Ascend-mindxsdk-hybrid-torchrec-${torchrec_version}-${pt_version}-linux-${ARCH}.tar.gz"
    if [ -f "${package_name}" ]; then
        rm "${package_name}"
    fi
    ls -l ./dist
    tar -czvf "${package_name}" -C dist .
}

function compile_all_pkg()
{
    cd "${SCRIPT_PATH}"
    ARCH=$(uname -m)
    torchrec_version=$(get_torchrec_ver)
    export BUILD_VERSION="${torchrec_version}"
    build_hybrid_torchrec
    build_torchrec_embcache
    archive_target_pkg
}

# 检查是否有pytorch 2.6.0python虚拟环境。若存在则基于每个虚拟环境编译。适用于流水线构建
[ -e /opt/buildtools/pt260_env/bin/activate ] && source /opt/buildtools/pt260_env/bin/activate && compile_all_pkg && deactivate pt260_env

# 默认构建一次。 在CI环境上是基于python默认环境的 pytorch 2.7.1版本再构建一次包
compile_all_pkg