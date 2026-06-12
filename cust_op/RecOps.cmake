# ===========================================================================
# RecOps.cmake — RecSDK AscendC 算子构建（风格对齐 FbgemmAscend.cmake）
# 新增算子只需修改下方算子列表，无需改动 CMakeLists.txt
# ===========================================================================

set(RECSDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ASCENDC_OPS_DIR "${RECSDK_SOURCE_DIR}/ascendc_op/ai_core_op")
set(ASCENDC_STAGE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/custom_opp)
set(ASCENDC_EXTRACT_SCRIPT "${RECSDK_SOURCE_DIR}/scripts/extract_custom_opp_runs.sh")
set(ASCENDC_STAGE_SUBDIRS "")
set_property(GLOBAL PROPERTY RECSDK_ASCEND_TARGETS "")

# AscendC 默认串行构建（更稳定）；可通过 -DRECSDK_ASCEND_SERIAL_BUILD=OFF 开启并行。
if(NOT DEFINED RECSDK_ASCEND_SERIAL_BUILD)
    set(RECSDK_ASCEND_SERIAL_BUILD ON)
endif()

# ================================ A5 ops ================================
set(RECSDK_CUSTOM_OPS_A5
    concat_jagged_tensor
    concat_jagged_tensor_grad
    disentangle_attention
    gather_for_rank1
    hstu_dense_backward
    hstu_dense_backward_fuxi
    hstu_dense_forward
    hstu_dense_forward_fuxi
    in_linear_silu
    in_linear_silu_backward
    index_select_for_rank1_backward
    ln_mul
    multislice_concat
    norm_multiply_dropout
    norm_multiply_dropout_backward
    relative_attn_bias_backward
    relative_attn_bias_pos
    relative_attn_bias_time
    reverse_sequence
    token_mixing
)

# ================================ A3 ops ================================
set(RECSDK_CUSTOM_OPS_A3
    concat_jagged_tensor
    concat_jagged_tensor_grad
    disentangle_attention
    gather_for_rank1
    hstu_dense_backward
    hstu_dense_backward_fuxi
    hstu_dense_forward
    hstu_dense_forward_fuxi
    in_linear_silu
    in_linear_silu_backward
    index_select_for_rank1_backward
    ln_mul
    # multislice_concat
    norm_multiply_dropout
    norm_multiply_dropout_backward
    relative_attn_bias_backward
    relative_attn_bias_pos
    relative_attn_bias_time
    reverse_sequence
    # token_mixing
)

# ================================ A2 ops ================================
set(RECSDK_CUSTOM_OPS_A2
    concat_jagged_tensor
    concat_jagged_tensor_grad
    disentangle_attention
    gather_for_rank1
    hstu_dense_backward
    hstu_dense_backward_fuxi
    hstu_dense_forward
    hstu_dense_forward_fuxi
    in_linear_silu
    in_linear_silu_backward
    index_select_for_rank1_backward
    ln_mul
    multislice_concat
    norm_multiply_dropout
    norm_multiply_dropout_backward
    relative_attn_bias_backward
    relative_attn_bias_pos
    relative_attn_bias_time
    reverse_sequence
    token_mixing
)

# ---------------------------------------------------------------------------
# AscendC 算子构建版本（受 CANN 平台信息限制，只能编译 CANN 支持的芯片）
# 默认构建全部芯片，可通过 RECSDK_BUILD_VERS 限制（如仅有 A2 CANN 时传 "A2,A3"）
# 注意：torch_plugin 适配层 .so 按芯片拆分编译，仅包含对应芯片的算子
# ---------------------------------------------------------------------------
if(NOT DEFINED RECSDK_BUILD_VERS OR RECSDK_BUILD_VERS STREQUAL "")
    set(RECSDK_BUILD_VERS "A2,A3,A5")
endif()
string(REPLACE "," ";" RECSDK_BUILD_VERS "${RECSDK_BUILD_VERS}")
string(REPLACE " " ";" RECSDK_BUILD_VERS "${RECSDK_BUILD_VERS}")

# ---------------------------------------------------------------------------
# 芯片 → (build_ver, ai_core) 映射
# ---------------------------------------------------------------------------
function(_recsdk_get_target_info variant out_build out_ai)
    if(variant STREQUAL "A5")
        set(${out_build} "c310" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend950" PARENT_SCOPE)
    elseif(variant STREQUAL "A2")
        set(${out_build} "v220" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend910B1" PARENT_SCOPE)
    elseif(variant STREQUAL "A3")
        set(${out_build} "v220" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend910_93" PARENT_SCOPE)
    elseif(variant STREQUAL "310P")
        set(${out_build} "v220" PARENT_SCOPE)
        set(${out_ai} "ai_core-Ascend310P3" PARENT_SCOPE)
    else()
        set(${out_build} "" PARENT_SCOPE)
        set(${out_ai} "" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# 单个算子构建函数
#   直接调用算子的 run.sh 脚本，传递 ai_core 参数
# ---------------------------------------------------------------------------
function(_recsdk_add_ascendc_op op_name build_ver ai_core stage_dir variant)
    if(NOT ai_core)
        message(WARNING "ASCENDC ai_core is empty; skipping ${op_name} (${variant})")
        return()
    endif()

    set(work_dir "${ASCENDC_OPS_DIR}/${op_name}/${build_ver}")
    if(NOT EXISTS "${work_dir}/run.sh")
        message(STATUS "Skipping ${op_name} (${variant}); run.sh not found in ${work_dir}")
        return()
    endif()

    set(stamp "${CMAKE_CURRENT_BINARY_DIR}/${op_name}_${variant}.stamp")
    set(target_name "ascendc_${op_name}_${variant}")

    # 在 CMake 配置阶段检测 run.sh 参数风格，避免在 make/bash -c 中被错误拆词。
    file(READ "${work_dir}/run.sh" _run_sh_content)
    string(FIND "${_run_sh_content}" "parse_arguments" _has_parse_arguments)
    if(_has_parse_arguments GREATER -1)
        set(run_args --ai-core ${ai_core})
    else()
        set(run_args ${ai_core})
    endif()

    add_custom_command(
        OUTPUT ${stamp}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${stage_dir}
        COMMAND bash ./run.sh ${run_args}
        COMMAND bash ${ASCENDC_EXTRACT_SCRIPT} ${op_name} ${stage_dir}
        COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
        WORKING_DIRECTORY ${work_dir}
        DEPENDS ${work_dir}/run.sh ${ASCENDC_EXTRACT_SCRIPT}
        COMMENT "Building AscendC ${op_name} (${variant})"
        )

    add_custom_target(${target_name} ALL DEPENDS ${stamp})

    # 同名算子在不同芯片变体间会落到同一个 build_ver 目录（例如 A2/A3 都是 v220），
    # 因此必须始终串行，避免多个 run.sh 同时进入同一工作目录导致竞争。
    get_property(_prev_target GLOBAL PROPERTY "RECSDK_PREV_${op_name}" SET)
    if(_prev_target)
        get_property(_prev_name GLOBAL PROPERTY "RECSDK_PREV_${op_name}")
        add_dependencies(${target_name} ${_prev_name})
    endif()
    set_property(GLOBAL PROPERTY "RECSDK_PREV_${op_name}" "${target_name}")

    # 全局串行 AscendC 构建：多个 run.sh 会读写共享模板目录，
    # 并行执行会导致随机失败（日志经常只显示顶层 Error 2）。
    if(RECSDK_ASCEND_SERIAL_BUILD)
        get_property(_last_target GLOBAL PROPERTY RECSDK_ASCEND_LAST_TARGET SET)
        if(_last_target)
            get_property(_last_name GLOBAL PROPERTY RECSDK_ASCEND_LAST_TARGET)
            add_dependencies(${target_name} ${_last_name})
        endif()
        set_property(GLOBAL PROPERTY RECSDK_ASCEND_LAST_TARGET "${target_name}")
    endif()
    set_property(GLOBAL APPEND PROPERTY RECSDK_ASCEND_TARGETS ${target_name})
endfunction()

# ---------------------------------------------------------------------------
# 遍历所有芯片变体，编译对应的算子
# ---------------------------------------------------------------------------
foreach(_variant ${RECSDK_BUILD_VERS})
    _recsdk_get_target_info(${_variant} _build_ver _ascendc_ai_core)
    if(NOT _build_ver OR NOT _ascendc_ai_core)
        message(WARNING "Unknown variant ${_variant}; skipping")
        continue()
    endif()

    if(_variant STREQUAL "A3")
        set(_transform_json "${RECSDK_SOURCE_DIR}/ascendc_op/config/transform.json")
        if(EXISTS "${_transform_json}")
            file(READ "${_transform_json}" _transform_content)
            string(FIND "${_transform_content}" "\"ascend910_93\"" _has_a3_mapping)
            if(_has_a3_mapping EQUAL -1)
                message(WARNING "A3 mapping key ascend910_93 not found in transform.json; skipping A3 build")
                continue()
            endif()
        else()
            message(WARNING "transform.json not found; skipping A3 build")
            continue()
        endif()
    endif()

    set(_stage_dir "${ASCENDC_STAGE_ROOT}/${_variant}")
    set(_vendors_for_config "")

    # 获取当前芯片的算子列表
    set(_chip_ops "${RECSDK_CUSTOM_OPS_${_variant}}")

    foreach(_op_name ${_chip_ops})
        _recsdk_add_ascendc_op(
            ${_op_name} ${_build_ver} ${_ascendc_ai_core}
            ${_stage_dir} ${_variant}
        )
        list(APPEND _vendors_for_config ${_op_name})
    endforeach()

    # 生成 vendors/config.ini（CANN 运行时用于发现和加载自定义算子）
    if(_vendors_for_config)
        string(REPLACE ";" "," _vendor_csv "${_vendors_for_config}")
        file(MAKE_DIRECTORY ${_stage_dir}/vendors)
        file(WRITE ${_stage_dir}/vendors/config.ini "load_priority=${_vendor_csv}\n")
        list(APPEND ASCENDC_STAGE_SUBDIRS ${_variant})
    endif()
endforeach()

list(REMOVE_DUPLICATES ASCENDC_STAGE_SUBDIRS)
get_property(ASCENDC_TARGETS GLOBAL PROPERTY RECSDK_ASCEND_TARGETS)

# ---------------------------------------------------------------------------
# C++ 适配层源文件（framework/torch_plugin），按芯片拆分
# 路径相对于 RECSDK_SOURCE_DIR
# ---------------------------------------------------------------------------
set(RECSDK_TORCH_LIBRARY_DIR framework/torch_plugin/torch_library)

set(RECSDK_ADAPTER_SRCS_A5
    ${RECSDK_TORCH_LIBRARY_DIR}/concat_2d_jagged/concat_jagged_tensor.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/disentangle_attention/DisentangleAttenFusion.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/gather_for_rank1/gather_for_rank1.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_dense.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_jagged.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_paged.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu_dense_backward_fuxi/HstuDenseNpuFusionFuxi.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu_dense_forward_fuxi/HstuDenseNpuFusionFuxi.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/in_linear_silu/in_linear_silu.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/ln_mul/ln_mul.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/multislice_concat/multislice_concat.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/norm_multiply_dropout/norm_multiply_dropout.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/relative_attn_bias/relative_attn_bias_pos.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/relative_attn_bias/relative_attn_bias_time.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/reverse_sequence/reverse_sequence.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/token_mixing/token_mixing.cpp
)

set(RECSDK_ADAPTER_SRCS_A3
    ${RECSDK_TORCH_LIBRARY_DIR}/concat_2d_jagged/concat_jagged_tensor.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/disentangle_attention/DisentangleAttenFusion.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/gather_for_rank1/gather_for_rank1.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_dense.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_jagged.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_paged.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu_dense_backward_fuxi/HstuDenseNpuFusionFuxi.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu_dense_forward_fuxi/HstuDenseNpuFusionFuxi.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/in_linear_silu/in_linear_silu.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/ln_mul/ln_mul.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/norm_multiply_dropout/norm_multiply_dropout.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/relative_attn_bias/relative_attn_bias_pos.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/relative_attn_bias/relative_attn_bias_time.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/reverse_sequence/reverse_sequence.cpp
)

set(RECSDK_ADAPTER_SRCS_A2
    ${RECSDK_TORCH_LIBRARY_DIR}/concat_2d_jagged/concat_jagged_tensor.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/disentangle_attention/DisentangleAttenFusion.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/gather_for_rank1/gather_for_rank1.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_dense.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_jagged.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu/hstu_paged.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu_dense_backward_fuxi/HstuDenseNpuFusionFuxi.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/hstu_dense_forward_fuxi/HstuDenseNpuFusionFuxi.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/in_linear_silu/in_linear_silu.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/ln_mul/ln_mul.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/multislice_concat/multislice_concat.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/norm_multiply_dropout/norm_multiply_dropout.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/relative_attn_bias/relative_attn_bias_pos.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/relative_attn_bias/relative_attn_bias_time.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/reverse_sequence/reverse_sequence.cpp
    ${RECSDK_TORCH_LIBRARY_DIR}/token_mixing/token_mixing.cpp
)
