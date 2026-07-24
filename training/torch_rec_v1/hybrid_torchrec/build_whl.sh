#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

SCRIPT_PATH=$(cd $(dirname $0); pwd)
TORCHREC_EMBCACHE_PATH="${SCRIPT_PATH}/../torchrec_embcache"

# ==============================================================================
# CCACHE 配置函数
# ==============================================================================
function setup_ccache()
{
    # 检查 ccache 是否安装
    if ! command -v ccache &> /dev/null; then
        echo "[INFO] ccache not found, using default compiler..."
        return 0
    fi

    echo "[INFO] ccache found, enabling compiler cache..."

    # 设置 ccache 路径到 PATH 最前面
    if [ -d "/usr/lib/ccache" ]; then
        export PATH=/usr/lib/ccache:$PATH
    elif [ -d "/usr/lib64/ccache" ]; then
        export PATH=/usr/lib64/ccache:$PATH
    else
        echo "Warning: ccache directory not found in standard paths."
    fi

    # 设置缓存目录和大小
    export CCACHE_DIR="${CCACHE_DIR:-/home/cache}"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-10G}"
    export CCACHE_COMPRESS=true
    export CCACHE_HASHDIR=true
    export CCACHE_SLOPPINESS="time_macros"
}

function get_pytorch_ver()
{
    local v260="2.6.0"
    local v271="2.7.1"
    local ver
    ver=$(python3 -m pip show torch 2>/dev/null | grep -E "^Version:" | awk '{print $2}') || {
        echo "ERROR: failed to get torch version !" >&2
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
    ver=$(python3 -m pip show torchrec 2>/dev/null | grep -E "^Version:" | awk '{print $2}')
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

function merge_compile_commands()
{
    local file1="${SCRIPT_PATH}/src/build/compile_commands.json"
    local file2="${TORCHREC_EMBCACHE_PATH}/src/cmake_build/compile_commands.json"
    local recsdk_root=$(cd "$SCRIPT_PATH/../../.." && pwd)
    local output="${recsdk_root}/compile_commands.json"
    local temp_file=$(mktemp)

    echo "[" > "$temp_file"

    if [ -f "$file1" ]; then
        sed '1d;$d' "$file1" >> "$temp_file"
    fi

    if [ -f "$file1" ] && [ -f "$file2" ]; then
        echo "," >> "$temp_file"
    fi

    if [ -f "$file2" ]; then
        sed '1d;$d' "$file2" >> "$temp_file"
    fi

    echo "]" >> "$temp_file"

    mv "$temp_file" "$output"

    local count1=$(if [ -f "$file1" ]; then sed '1d;$d' "$file1" | grep -c '^\s*{' || echo 0; else echo 0; fi)
    local count2=$(if [ -f "$file2" ]; then sed '1d;$d' "$file2" | grep -c '^\s*{' || echo 0; else echo 0; fi)
    local total=$((count1 + count2))

    echo "Merged hybrid_torchrec and torchrec_embcache compile_commands to ${output}"
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
    merge_compile_commands
}

# ==============================================================================
# 初始化 CCACHE
# ==============================================================================
setup_ccache
# 检查是否有pytorch 2.6.0python虚拟环境。若存在则基于每个虚拟环境编译。适用于流水线构建
[ -e /opt/buildtools/pt260_env/bin/activate ] && source /opt/buildtools/pt260_env/bin/activate && compile_all_pkg && deactivate pt260_env

# 默认构建一次。 在CI环境上是基于python默认环境的 pytorch 2.7.1版本再构建一次包
compile_all_pkg
