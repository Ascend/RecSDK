#!/bin/bash
# ==============================================================================
# 【新版·CMake 封装】hstu_attn_metadata (AI CPU 自定义算子) 便捷构建脚本（薄封装）
#
# 另有旧版手写编译脚本 run_manual.sh（不依赖 CMake），暂时保留以备对照。
#
# 所有编译/链接/打包逻辑已迁移到 CMakeLists.txt，本脚本仅负责：
#   1. 探测 / source CANN 环境（ASCEND_HOME_PATH）
#   2. 调用 cmake 配置、构建、安装、运行
#
# 用法:
#   bash run.sh                 # build + install + run example (默认)
#   bash run.sh --stage=build   # 仅配置并编译 (kernel + opapi + 打包 vendor)
#   bash run.sh --stage=install # 编译并安装到 opp/vendors
#   bash run.sh --stage=run     # 编译并运行 C++ 冒烟 example
#
# 直接用 cmake 亦可（等价）:
#   cmake -B build && cmake --build build -j        # = --stage=build
#   cmake --install build                            # = 安装
#   cmake --build build --target run_example         # = 运行 example
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

STAGE="all"
for arg in "$@"; do
    case "${arg}" in
        --stage=*) STAGE="${arg#*=}" ;;
        *) echo "[WARN] unknown arg: ${arg}" ;;
    esac
done

# ------------------------------------------------------------------------------
# 环境探测：确保 ASCEND_HOME_PATH 可用
# ------------------------------------------------------------------------------
if [ -z "${ASCEND_HOME_PATH}" ]; then
    for c in /usr/local/Ascend/ascend-toolkit/set_env.sh \
             /usr/local/Ascend/cann-A2/cann-9.0.0/../set_env.sh; do
        [ -f "${c}" ] && source "${c}" && break
    done
fi
if [ -z "${ASCEND_HOME_PATH}" ] || [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置或不可用, 请先 source set_env.sh 后重试"; exit 1
fi
ASCEND_HOME="${ASCEND_HOME_PATH}"
echo "[INFO] ASCEND_HOME_PATH = ${ASCEND_HOME}"

# ------------------------------------------------------------------------------
# cmake 配置 + 构建（kernel + opapi + 打包 vendor 均在 CMakeLists 中完成）
# ------------------------------------------------------------------------------
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DASCEND_HOME_PATH="${ASCEND_HOME}"
cmake --build "${BUILD_DIR}" -j

case "${STAGE}" in
    build)
        : ;;   # 已完成编译 + 打包
    install)
        cmake --install "${BUILD_DIR}" ;;
    run)
        cmake --build "${BUILD_DIR}" --target run_example ;;
    all)
        cmake --install "${BUILD_DIR}"
        cmake --build "${BUILD_DIR}" --target run_example ;;
    *)
        echo "[ERROR] unknown stage: ${STAGE}"; exit 1 ;;
esac
echo "[DONE] stage=${STAGE}"
