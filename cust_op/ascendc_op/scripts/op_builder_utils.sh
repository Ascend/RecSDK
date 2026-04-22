#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 防止被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This file is a library. Please source it."
    exit 1
fi

# ==============================================================================
# 路径初始化逻辑
# 基于本脚本位置确定 __PROJECT_ROOT 和 CONFIG_DIR
# ==============================================================================

# 获取本脚本的绝对路径
readonly __UTILS_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
# 获取本脚本所在目录 (project_root/scripts)
readonly __UTILS_DIR="$(dirname "${__UTILS_SCRIPT_PATH}")"
# 推导项目根目录 (project_root) - 假设 scripts 在根目录下
readonly __PROJECT_ROOT="$(dirname "${__UTILS_DIR}")"
# 推导 config 目录
readonly CONFIG_DIR="${__PROJECT_ROOT}/config"
# ONNIX 适配层路径
readonly ONNX_PATH="${__PROJECT_ROOT}/build/scripts/onnx_plugin"
readonly JSON_FILE="${ONNX_PATH}/json.hpp"

# ==============================================================================
# 参数解析
# 用法: parse_arguments "$@"
# 注意: vendor_name 需要由调用者传入
# ==============================================================================
parse_arguments() {
    case "${AI_CORE_PROFILE:-v220}" in
        c310)
            : "${ai_core:=ai_core-Ascend950}"
            ;;
        v220)
            : "${ai_core:=ai_core-Ascend910B1}"
            ;;
    esac

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --vendor-name)
                if [ -z "$2" ]; then
                    echo "ERROR: --vendor-name requires a non-empty argument." >&2
                    return 1
                fi
                vendor_name="$2"
                shift 2
                ;;
            --ai-core)
                ai_core="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter passed: $1" >&2
                return 1
                ;;
        esac
    done

    if [ -z "$vendor_name" ]; then
        echo "ERROR: --vendor-name is required." >&2
        return 1
    fi

    export vendor_name
    export ai_core
    return 0
}

# ==============================================================================
# 验证 AI Core
# ==============================================================================

VALID_AI_CORES=(
    "ai_core-Ascend910B1"
    "ai_core-Ascend910B2"
    "ai_core-Ascend910B3"
    "ai_core-Ascend910B4"
    "ai_core-Ascend910_93"
    "ai_core-Ascend310P3"
)

VALID_AI_CORES_C310=(
    "ai_core-Ascend950"
)

validate_ai_core() {
    local target_core="$1"

    local profile="${AI_CORE_PROFILE:-v220}"
    case "$profile" in
        c310)
            for valid_core in "${VALID_AI_CORES_C310[@]}"; do
                if [ "$target_core" = "$valid_core" ]; then
                    echo "ai_core $target_core"
                    return 0
                fi
            done
            echo "Error: ai core must be one of: [${VALID_AI_CORES_C310[*]}] (AI_CORE_PROFILE=${profile})" >&2
            return 1
            ;;
        v220)
            for valid_core in "${VALID_AI_CORES[@]}"; do
                if [ "$target_core" = "$valid_core" ]; then
                    echo "ai_core $target_core"
                    return 0
                fi
            done
            echo "Error: ai core must be one of: [${VALID_AI_CORES[*]}] (AI_CORE_PROFILE=${profile})" >&2
            return 1
            ;;
        *)
            echo "ERROR: AI_CORE_PROFILE must be 'v220' or 'c310' (or unset for v220), got: '${AI_CORE_PROFILE}'" >&2
            return 1
            ;;
    esac
}

# ==============================================================================
# 获取系统信息
# 用法: check_system_and_cann "$target_core"
# ==============================================================================

# 获取g++路径
get_gpp_path() {
    local gpp_path
    gpp_path=$(which g++)
    if [ -z "$gpp_path" ]; then
        echo "ERROR: g++ not found in PATH." >&2
        return 1
    fi
    export GPP_PATH="$gpp_path"
    return 0
}

check_system_and_cann() {
    local target_core="$1"

    # 获取架构
    ARCH=$(uname -m)
    if [ -z "${ARCH}" ]; then
        echo "ERROR: get arch failed" >&2
        return 1
    fi

    # 查看cann版本
    local toolkit_path=$ASCEND_TOOLKIT_HOME
    if [ ! -d "$toolkit_path" ]; then
        echo "ERROR: cann not found at $toolkit_path" >&2
        return 1
    fi

    local info_file="${toolkit_path}/${ARCH}-linux/ascend_toolkit_install.info"
    if [ ! -f "$info_file" ]; then
        echo "ERROR: cann install info file not found" >&2
        return 1
    fi

    CANN_VERSION=$(grep "^version" "$info_file" | awk -F= '{print $2}')
    if [ -z "$CANN_VERSION" ]; then
        echo "ERROR: failed to parse cann version" >&2
        return 1
    fi

    echo "cann version: ${CANN_VERSION}"
    MAJOR_VERSION=$(echo "${CANN_VERSION}" | cut -d. -f1)
    echo "cann major version: ${MAJOR_VERSION}"

    # 获取系统ID
    OS_ID=$(cat /etc/os-release | sed -n 's/^ID=//p' | sed 's/^"//;s/"$//')
    if [ -z "${OS_ID}" ]; then
        echo "ERROR: get os_id failed" >&2
        exit 1
    fi

    # 只允许字母/数字/点/下划线/连字符（覆盖常见 os_id 与 arch）
    SAFE_REGEX='^[A-Za-z0-9._-]+$'
    if ! [[ "$OS_ID" =~ $SAFE_REGEX ]]; then
        echo "ERROR: invalid os_id: $OS_ID" >&2
        exit 1
    fi
    if ! [[ "$ARCH" =~ $SAFE_REGEX ]]; then
        echo "ERROR: invalid arch: $ARCH" >&2
        exit 1
    fi

    get_gpp_path || return 1

    export ARCH
    export CANN_VERSION
    export MAJOR_VERSION
    export OS_ID="$OS_ID"
    return 0
}

# ==============================================================================
# 用目标文件夹覆盖源目录
# 用法: overwrite_source_with_target "$source_dir" "$target_dir"
# ==============================================================================
overwrite_source_with_target() {
    local src_dir="$1"
    local tgt_dir="$2"

    if [ ! -d "$src_dir" ]; then
        echo "ERROR: Source directory not found: $src_dir" >&2
        return 1
    fi
    if [ ! -d "$tgt_dir" ]; then
        echo "ERROR: Target directory not found: $tgt_dir" >&2
        return 1
    fi

    find "$src_dir" -type f | while read -r src_file; do
        local relative_path="${src_file#$src_dir/}"
        local target_file="$tgt_dir/$relative_path"

        if [ -f "$target_file" ]; then
            cp -f "$target_file" "$src_file"
            echo "Overwrite $src_file with $target_file"
        else
            echo "WARNING: Target file not found for $src_file, skipping overwrite." >&2
        fi
    done
}

# ==============================================================================
# 根据 vendor_name 生成op_name
# ==============================================================================
generate_op_name() {
    local vendor_name="$1"
    # 将 vendor_name 转换为 PascalCase 作为 op_name
    local op_name=$(echo "$vendor_name" | sed -r 's/(^|_)([a-z])/\U\2/g')
    echo "$op_name"
}

# =============================================================================
# 生成算子代码
# 用法: gen_build_dir "$work_dir" "$vendor_name" "$op_name"
# ============================================================================
gen_build_dir() {
    local work_dir="$1"
    local vendor_name="$2"
    local op_name="$3"
    if [ -z "$op_name" ]; then
        op_name=$(generate_op_name "$vendor_name")
    fi
    rm -rf "${work_dir}/${vendor_name}"
    local json_file="${OPERATOR_JSON_FILE:-${work_dir}/${vendor_name}.json}"
    msopgen gen -i "${json_file}" -f tf -c ${ai_core} -lan cpp -out "${work_dir}/${vendor_name}" -m 0 -op ${op_name}
    if [ -d "${work_dir}/${vendor_name}/cmake" ] && [ "${MAJOR_VERSION}" -eq 9 ]; then
        export MAJOR_VERSION=8
    fi

    if [ "${MAJOR_VERSION}" -ge 9 ]; then
        overwrite_source_with_target "${work_dir}/${vendor_name}" \
        "${__PROJECT_ROOT}/ai_core_op/custom_op_template" || return 1
    fi
}

# ==============================================================================
# 替换算子工程源文件
# 用法: replace_operator_sources "$source_dir" "$target_dir"
# 可选环境变量 COPY_KERNEL_COMMON_UTILS=1: 在复制 op 源码后拷贝 kernel_common_utils.h
#   默认目录: ${__PROJECT_ROOT}/ai_core_op/common（即 cust_op/ascendc_op/ai_core_op/common）
#   覆盖: 设置 KERNEL_COMMON_UTILS_DIR 为含该头文件的目录路径
# ==============================================================================
replace_operator_sources() {
    local src_dir="$1"
    local tgt_dir="$2"

    if [ ! -d "$src_dir" ]; then
        echo "ERROR: Source directory not found: $src_dir" >&2
        return 1
    fi
    if [ ! -d "$tgt_dir" ]; then
        echo "ERROR: Target directory not found: $tgt_dir" >&2
        return 1
    fi

    # 清理旧文件 (防止残留)
    rm -rf "${tgt_dir}/op_kernel"/*.h "${tgt_dir}/op_kernel"/*.cpp 2>/dev/null || true
    rm -rf "${tgt_dir}/op_host"/*.h "${tgt_dir}/op_host"/*.cpp 2>/dev/null || true

    # 复制新文件
    cp -rf ${src_dir}/op_kernel/* "${tgt_dir}/op_kernel/"
    cp -rf ${src_dir}/op_host/* "${tgt_dir}/op_host/"

    if [ "${COPY_KERNEL_COMMON_UTILS}" = "1" ]; then
        cp -f "${KERNEL_COMMON_UTILS_DIR:-${__PROJECT_ROOT}/ai_core_op/common}/kernel_common_utils.h" \
            "${tgt_dir}/op_kernel/"
    fi
    return 0
}

# ==============================================================================
# 构建 ONNX 适配层
# 用法: build_onnx_adapter "$ai_core" "$onnx_path" "$json_file" "$vendor_name" "$work_dir"
# ==============================================================================
build_onnx_adapter() {
    local a_core="$1"
    local o_path="$2"
    local j_file="$3"
    local v_name="$4"
    local work_dir="$5"

    if [ "$a_core" = "ai_core-Ascend310P3" ]; then
        echo "Building ONNX adapter for ${a_core}..."
        local build_script="${o_path}/build_onnx.sh"

        if [ ! -f "$build_script" ]; then
            echo "ERROR: build_onnx.sh not found in ${o_path}" >&2
            return 1
        fi

        # 执行 build_onnx.sh
        bash "$build_script"

        local dest_dir="${work_dir}/${v_name}/framework/onnx_plugin"
        mkdir -p "$dest_dir"

        cp -rf "${j_file}" "$dest_dir"

        local src_onnx_dir="$(dirname "$work_dir")/onnx_plugin"
        if [ -d "$src_onnx_dir" ]; then
            cp -rf "${src_onnx_dir}"/* "$dest_dir"
        else
            echo "ERROR: onnx_plugin directory not found at $src_onnx_dir" >&2
            return 1
        fi
    fi
    return 0
}

# ==============================================================================
# 修改 CMakePresets.json
# 用法: configure_cmake_presets "$vendor_name" "$ai_core" "$major_version" "$target_dir" "$enable_catlass"(可选)
# target_dir: 即 ${work_dir}/${vendor_name} 的绝对路径
# c310 + OPERATOR_JSON_FILE：结束前复制到 dirname(target_dir)/${首参 v_name}.json，供 CPack 与 msopgen -i 共用同一描述文件。
# ==============================================================================
configure_cmake_presets() {
    local v_name="$1"
    local a_core="$2"
    local maj_ver="$3"
    local target_dir="$4"
    if [ -z "$5" ]; then
        local enable_catlass=${enable_catlass:-"False"}
    else
        local enable_catlass="$5"
    fi

    local cmake_file="${target_dir}/CMakePresets.json"

    if [ ! -f "$cmake_file" ]; then
        echo "ERROR: CMakePresets.json file not exist in ${target_dir}." >&2
        return 1
    fi

    # 修改 CANN 路径
    if [ "$maj_ver" -lt 9 ]; then
        sed -i "s:\"/usr/local/Ascend/latest\":\"${ASCEND_TOOLKIT_HOME}\":g" "$cmake_file"
    else
        sed -i "s:\"/usr/local/Ascend/cann\":\"${ASCEND_TOOLKIT_HOME}\":g" "$cmake_file"
    fi

    # 修改 vendor_name
    sed -i "s:\"customize\":\"${v_name}\":g" "$cmake_file"

    # 映射 Chip Name
    local chip_name_raw
    chip_name_raw=$(echo "$a_core" | sed 's/^ai_core-//i')
    local chip_name
    chip_name=$(echo "$chip_name_raw" | tr '[:upper:]' '[:lower:]')

    # 使用全局推导的 CONFIG_DIR
    local config_file="${CONFIG_DIR}/transform.json"

    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file transform.json not found at $config_file" >&2
        return 1
    fi

    local mapped_value
    mapped_value=$(python3 -c "import json; cfg=json.load(open('${config_file}')); \
                   print(cfg['SOC_VERSION_TYPE'].get('${chip_name}', ''))" 2>/dev/null)

    if [ -z "$mapped_value" ]; then
        echo "WARNING: No mapping found for chip '$chip_name' in $config_file." >&2
        return 1
    else
        echo "Mapping found: '$chip_name' -> '$mapped_value'"
        sed -i "s:\"__ASCEND_COMPUTE_UNIT__\":\"${mapped_value}\":g" "$cmake_file"
    fi

    # 修改CMAKE_CROSS_PLATFORM_COMPILER
    sed -i "s:\"/usr/bin/aarch64-linux-gnu-g++\":\"${GPP_PATH}\":g" "$cmake_file"

    # 修改CATLASS_HOME以及ENABLE_CATLASS，需要判断是否有CATLASS_HOME环境变量
    if [ -n "$CATLASS_HOME" ] && [ "$enable_catlass" = "True" ]; then
        sed -i "s:\"/usr/local/Ascend/catlass\":\"${CATLASS_HOME}\":g" "$cmake_file"
        line=`awk '/ENABLE_CATLASS/{print NR}' "$cmake_file"`
        line=`expr ${line} + 2`
        sed -i "${line}s/False/True/g" "$cmake_file"
    fi

    # c310：op_kernel 里 install(FILES .../../../${v_name}.json) 实际解析到 dirname(target_dir)
    # （即与 run.sh 同级的 c310 目录），与 msopgen -i 常用路径（如 ../v220/*.json）不一致。
    # 已在各算子设置 OPERATOR_JSON_FILE 时，顺带复制一份供 CPack install 使用。
    if [ "${AI_CORE_PROFILE:-v220}" = "c310" ] && [ -n "${OPERATOR_JSON_FILE:-}" ] \
        && [ -f "${OPERATOR_JSON_FILE}" ]; then
        local c310_install_root
        c310_install_root="$(dirname "${target_dir}")"
        cp -f "${OPERATOR_JSON_FILE}" "${c310_install_root}/${v_name}.json"
    fi

    if [[ "$maj_ver" -lt 9 && ("$v_name" == "mxrec_sgd" || "$v_name" == "mxrec_fused_lazy_adam") ]]; then
            # 增加LOG_CPP编译选项支持错误日志打印
            sed -i '1 i include("${target_dir}"/../../../cmake/func.cmake)' "${target_dir}"/op_host/CMakeLists.txt

            line1=`awk '/target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)/{print NR}' "${target_dir}"/op_host/CMakeLists.txt`
            sed -i "${line1}s/OP_TILING_LIB/OP_TILING_LIB LOG_CPP/g" "${target_dir}"/op_host/CMakeLists.txt

            line2=`awk '/target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)/{print NR}' "${target_dir}"/op_host/CMakeLists.txt`
            sed -i "${line2}s/OP_PROTO_LIB/OP_PROTO_LIB LOG_CPP/g" "${target_dir}"/op_host/CMakeLists.txt

            sed -i '/\${ASCEND_CANN_PACKAGE_PATH}\/include/a\
            \${ASCEND_CANN_PACKAGE_PATH}\/pkg_inc
            ' "${target_dir}"/cmake/*.cmake
        fi

    return 0
}

# ==============================================================================
# 为 op_kernel/CMakeLists.txt 追加额外 AscendC 编译选项（新/旧 CMake 二选一）
# 用法: apply_op_kernel_compile_options_dual "$target_dir" "$opts_body"
#   opts_body: 同 add_ops_compile_options 里 OPTIONS 后的内容，如 -DENABLE_CV_COMM_VIA_SSBUF=true
# 有 npu_op_kernel_options 则把 opts_body 接到首行 OPTIONS "--cce-long-call=true" 后；否则首行插入 add_ops_compile_options。
# maj<9 时 prepare_and_build 会再插一行仅含 --cce-long-call=true，opts_body 一般勿重复写 long-call。
# ==============================================================================
apply_op_kernel_compile_options_dual() {
    local target_dir="$1"
    local opts_body="$2"
    local f="${target_dir}/op_kernel/CMakeLists.txt"
    local tmp

    if [ ! -f "$f" ]; then
        echo "ERROR: op_kernel/CMakeLists.txt not found under ${target_dir}" >&2
        return 1
    fi
    if [ -z "$opts_body" ]; then
        echo "ERROR: apply_op_kernel_compile_options_dual: opts_body is empty" >&2
        return 1
    fi

    if grep -q 'npu_op_kernel_options' "$f"; then
        if ! grep -q 'OPTIONS "--cce-long-call=true"' "$f"; then
            echo "ERROR: ${f} uses npu_op_kernel_options but lacks OPTIONS \"--cce-long-call=true\" (sync with custom_op_template)" >&2
            return 1
        fi
        tmp=$(mktemp "${TMPDIR:-/tmp}/op_builder_utils.XXXXXX")
        awk -v extra=" ${opts_body}" '
            /OPTIONS "--cce-long-call=true"/ && done == 0 {
                sub(/OPTIONS "--cce-long-call=true"/, "&" extra)
                done = 1
            }
            { print }
        ' "$f" > "$tmp" && mv "$tmp" "$f"
    else
        tmp=$(mktemp "${TMPDIR:-/tmp}/op_builder_utils.XXXXXX")
        {
            printf '%s\n' "add_ops_compile_options(ALL OPTIONS ${opts_body})"
            cat "$f"
        } > "$tmp" && mv "$tmp" "$f"
    fi
    return 0
}

# ==============================================================================
# 编译前准备与执行编译，兼容cann 9.0以下版本
# 用法: prepare_and_build "$major_version" "$vendor_name" "$target_dir" "$enable_catlass"(可选)
# ==============================================================================
prepare_and_build() {
    local maj_ver="$1"
    local v_name="$2"
    local target_dir="$3"
    if [ -z "$4" ]; then
        local enable_catlass=${enable_catlass:-"False"}
    else
        local enable_catlass="$4"
    fi

    if [ "$maj_ver" -lt 9 ]; then
        echo "Preparing legacy build environment (CANN < 9.0)..."

        # 禁止 CRC
        local makeself_cmake="${target_dir}/cmake/makeself.cmake"
        if [ -f "$makeself_cmake" ]; then
            sed -i 's/--nomd5/--nomd5 --nocrc/g' "$makeself_cmake"
        fi

        # 修改 op_kernel/CMakeLists.txt
        local kernel_cmakelists="${target_dir}/op_kernel/CMakeLists.txt"
        if [ -f "$kernel_cmakelists" ]; then
            local add_line="install(FILES \${CMAKE_CURRENT_SOURCE_DIR}/../../${v_name}.json \
                            DESTINATION packages/vendors/\${vendor_name}/op_impl/ai_core/tbe/\${v_name}_impl/dynamic)"
            sed -i "\$a\\$add_line" "$kernel_cmakelists"
            sed -i '1i\add_ops_compile_options(ALL OPTIONS --cce-long-call=true)' "$kernel_cmakelists"
        fi

        if [ -n "$CATLASS_HOME" ] && [ "$enable_catlass" = "True" ] && [ -f "$kernel_cmakelists" ]; then
            local catlass_include_dir="${CATLASS_HOME}/include"
            if [ "${AI_CORE_PROFILE:-v220}" = "c310" ]; then
                sed -i "2i\add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=3510 -DARCH_CODE=Ascend950 -DUSE_TLA=1 \
                    -DCATLASS_BISHENG_ARCH=a5 -DCATLASS_ARCH_A5_ENABLED=true -DENABLE_CV_COMM_VIA_SSBUF=true \
                    -DCATLASS_HOME=${CATLASS_HOME} -I${catlass_include_dir})" "${kernel_cmakelists}"
            else
                sed -i "2i\add_ops_compile_options(ALL OPTIONS -DCATLASS_ARCH=2201 -DARCH_CODE=AtlasA2 -DUSE_TLA=0 \
                    -DCATLASS_BISHENG_ARCH=a2 -DCATLASS_ARCH_A2_ENABLED=true -DENABLE_CV_COMM_VIA_SSBUF=true \
                    -DCATLASS_HOME=${CATLASS_HOME} -I${catlass_include_dir})" "${kernel_cmakelists}"
            fi
        fi

        # 修改 op_host/CMakeLists.txt
        local host_cmakelists="${target_dir}/op_host/CMakeLists.txt"
        if [ -f "$host_cmakelists" ]; then
            sed -i "1 i include(../../../../cmake/func.cmake)" "$host_cmakelists"

            local line1
            line1=$(awk '/target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)/{print NR}' "$host_cmakelists")
            if [ -n "$line1" ]; then
                sed -i "${line1}s/OP_TILING_LIB/OP_TILING_LIB LOG_CPP/g" "$host_cmakelists"
            fi

            local line2
            line2=$(awk '/target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)/{print NR}' "$host_cmakelists")
            if [ -n "$line2" ]; then
                sed -i "${line2}s/OP_PROTO_LIB/OP_PROTO_LIB LOG_CPP/g" "$host_cmakelists"
            fi
        fi

        # 修改 cmake/*.cmake
        if compgen -G "${target_dir}/cmake/*.cmake" > /dev/null; then
            for f in "${target_dir}"/cmake/*.cmake; do
                sed -i '/\${ASCEND_CANN_PACKAGE_PATH}\/include/a\
                        \${ASCEND_CANN_PACKAGE_PATH}\/pkg_inc
                        ' "$f"
            done
        fi
    fi

    # 执行编译 (需要在 target_dir 下运行 build.sh)
    # 由于 build.sh 内部可能有相对路径依赖，这里必须 cd 进去执行，但用子 shell 包裹，不影响主环境
    local build_script="${target_dir}/build.sh"
    if [ -f "$build_script" ]; then
        (
            cd "$target_dir" || exit 1
            bash build.sh
        ) || return 1
    else
        echo "ERROR: build.sh not found in ${target_dir}" >&2
        return 1
    fi

    return 0
}

# ==============================================================================
# 辅助函数: 安装算子包
# 用法: install_operator_package "$os_id" "$arch" "$target_dir"
# ==============================================================================
install_operator_package() {
    local o_id="$1"
    local a_arch="$2"
    local target_dir="$3"

    local installer="${target_dir}/build_out/custom_opp_${o_id}_${a_arch}.run"

    if [ ! -f "$installer" ]; then
        echo "ERROR: Installer package not found: $installer" >&2
        return 1
    fi

    echo "Installing operator package: $installer"
    bash -- "$installer"
    return $?
}

# ==============================================================================
# 整体构建流程
# 注意: vendor_name 需要由调用者传入，with_onnx 需要由调用者传入（如果需要构建onnx适配层则传入 "true"）
# 使用说明：
#  1. 该函数是构建算子的主流程，依次调用前面定义的各个步骤函数。
#  2. 使用该函数时，需要在调用脚本里进行source ${__UTILS_SCRIPT_PATH}，然后调用 build_and_install_operator 函数，并传入必要的参数。
#  3. 如需要对相应的步骤或者变量进行定制，可以在调用脚本里覆盖相应的函数或者变量，然后再调用 build_and_install_operator。
#  4. 执行顺序为，先 source，按需设置环境变量，再调用 build_and_install_operator。
#  5. 参数： "$work_dir" "$vendor_name" "$with_onnx"(可选，需为字面值 true 才构建 ONNX)
# 可选环境变量（与单步脚本手写链一致）:
#   OPERATOR_JSON_FILE / OPERATOR_SOURCE_ROOT / INSERT_SUPPORT_950_PATHS / COPY_KERNEL_COMMON_UTILS
#   CMAKE_PRESET_VENDOR_NAME — CMake customize 与 prepare_and_build 第二参（工程目录仍为 vendor_name，如 mxrec_*）
#   MSOPGEN_OP_NAME — 传给 gen_build_dir 第三参；不设则按 vendor_name 推导 PascalCase（fused 需设 LazyAdam/Sgd）
#   enable_catlass — True/False；build_and_install_operator 未传第 4/5 步参数时由 configure_cmake_presets / prepare_and_build 读取（默认 False）
# ==============================================================================
build_and_install_operator() {
    local work_dir="$1"
    local vendor_name="$2"
    local with_onnx="$3"
    local cmake_preset="${CMAKE_PRESET_VENDOR_NAME:-$vendor_name}"

    echo "=========================================="
    echo "Start Building Operator: ${vendor_name}"
    echo "Target AI Core: ${ai_core}"
    echo "Work Directory : ${work_dir}"
    echo "=========================================="

    validate_ai_core "$ai_core" || return 1
    check_system_and_cann "$ai_core" || return 1
    gen_build_dir "$work_dir" "$vendor_name" "${MSOPGEN_OP_NAME:-}" || return 1
    local op_src_root="${OPERATOR_SOURCE_ROOT:-${work_dir}}"
    replace_operator_sources "$op_src_root" "${work_dir}/${vendor_name}" || return 1
    if [ -n "${INSERT_SUPPORT_950_PATHS:-}" ]; then
        local _r
        for _r in ${INSERT_SUPPORT_950_PATHS}; do
            sed -i "1i #define SUPPORT_950" "${work_dir}/${vendor_name}/${_r}"
        done
    fi
    if [ "$with_onnx" = "true" ]; then
        build_onnx_adapter "$ai_core" "$ONNX_PATH" "$JSON_FILE" "$vendor_name" "$work_dir" || return 1
    fi
    configure_cmake_presets "$cmake_preset" "$ai_core" "$MAJOR_VERSION" "${work_dir}/${vendor_name}" || return 1
    prepare_and_build "$MAJOR_VERSION" "$cmake_preset" "${work_dir}/${vendor_name}" || return 1
    install_operator_package "$OS_ID" "$ARCH" "${work_dir}/${vendor_name}" || return 1

    echo "=========================================="
    echo "Operator ${vendor_name} built and installed successfully."
    echo "=========================================="
    return 0
}