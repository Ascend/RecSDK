# ==============================================================================
# hstu_attn_metadata (AI CPU 自定义算子) 独立 CMake 工程
#
# 该算子从 ops-transformer 移植而来，是一个纯 AI CPU 算子，不依赖 ops-transformer
# 构建框架。本 CMake 产出并打包：
#   1. libtransformer_aicpu_kernels.so  —— AICPU kernel（aarch64 交叉编译，CANN hcc 工具链）
#   2. libcust_opapi.so                 —— aclnn 对外接口（host 编译）
#   3. build/vendor/<VENDOR_NAME>/       —— CANN 自定义算子 vendor 目录（cmake --build 产出）
#   4. cmake --install                   —— 安装到 opp/vendors 并合并 config.ini
#
# 重要约束（CANN 9.0.0 实测）：
#   * VENDOR_NAME 必须以 `_transformer` 结尾，否则 AICPU 注册表白名单校验不过
#     （日志 "suffix[xxx] is not in customer white list"），运行时报 107000。
#   * kernelSo 名必须与 aicpu json 中的 kernelSo 一致（libtransformer_aicpu_kernels.so）。
#   * 运行时必须 export ASCEND_CUSTOM_OPP_PATH 指向 vendor 目录。
# ==============================================================================
cmake_minimum_required(VERSION 3.14)
project(hstu_attn_metadata CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# 基本配置
# ------------------------------------------------------------------------------
set(OP_NAME       hstu_attn_metadata)
set(VENDOR_NAME   hstu_attn_metadata_transformer)   # 后缀必须为
set(KERNEL_SO     libtransformer_aicpu_kernels.so)

# ------------------------------------------------------------------------------
# 定位 CANN
# ------------------------------------------------------------------------------
if(NOT DEFINED ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
  elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME_PATH $ENV{ASCEND_TOOLKIT_HOME})
  else()
    set(ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest)
  endif()
endif()
if(NOT IS_DIRECTORY "${ASCEND_HOME_PATH}")
  message(FATAL_ERROR "CANN 路径未找到：${ASCEND_HOME_PATH}，请先 source set_env.sh 或 -DASCEND_HOME_PATH=")
endif()

set(_arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(_arch STREQUAL "")
  set(_arch "aarch64")
endif()
set(ARCH_DIR "${ASCEND_HOME_PATH}/${_arch}-linux")
if(NOT IS_DIRECTORY "${ARCH_DIR}")
  set(ARCH_DIR "${ASCEND_HOME_PATH}")
endif()
set(LIB64 "${ARCH_DIR}/lib64")

# opp/vendors 安装目录
if(DEFINED ENV{ASCEND_OPP_PATH})
  set(OPP_VENDORS "$ENV{ASCEND_OPP_PATH}/vendors")
else()
  set(OPP_VENDORS "${ASCEND_HOME_PATH}/opp/vendors")
endif()

# AICPU kernel 交叉编译器（CANN hcc 工具链）
find_program(CROSS_CXX
  NAMES aarch64-target-linux-gnu-g++
  PATHS "${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin"
  NO_DEFAULT_PATH)
if(NOT CROSS_CXX)
  find_program(CROSS_CXX NAMES aarch64-linux-gnu-g++ aarch64-target-linux-gnu-g++)
endif()
if(NOT CROSS_CXX)
  message(FATAL_ERROR "未找到 AICPU 交叉编译器 aarch64-target-linux-gnu-g++")
endif()

set(VENDOR_STAGE "${CMAKE_BINARY_DIR}/vendor/${VENDOR_NAME}")

message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")
message(STATUS "ARCH_DIR         = ${ARCH_DIR}")
message(STATUS "CROSS_CXX        = ${CROSS_CXX}")
message(STATUS "OPP_VENDORS      = ${OPP_VENDORS}")

# ==============================================================================
# 1. AICPU kernel（交叉编译，专用工具链 → 用 custom_command 驱动）
# ==============================================================================
set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/${OP_NAME}_aicpu.cpp")
set(KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${OP_NAME}_aicpu.cpp.o")
set(KERNEL_OUT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_SO}")

file(GLOB_RECURSE KERNEL_DEP_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/deps/attention_common/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/deps/hstu_attn/*.h")

set(KERNEL_INCLUDES
  -I${ARCH_DIR}/pkg_inc/aicpu_common/context
  -I${ARCH_DIR}/pkg_inc/aicpu_common/context/common
  -I${ARCH_DIR}/pkg_inc/aicpu_common/context/cpu_proto
  -I${ARCH_DIR}/pkg_inc/aicpu_common/context/utils
  -I${ARCH_DIR}/pkg_inc/aicpu
  -I${ARCH_DIR}/include
  -I${ASCEND_HOME_PATH}/include
  -I${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu
  -I${CMAKE_CURRENT_SOURCE_DIR}/deps/attention_common/op_kernel
  -I${CMAKE_CURRENT_SOURCE_DIR}/deps/hstu_attn
  -isystem${ASCEND_HOME_PATH}/pkg_inc/base)

set(KERNEL_DEFS
  -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1 -Dgoogle=ascend_private -DDISABLE_COMPILE_V1)

add_custom_command(
  OUTPUT "${KERNEL_OUT}"
  COMMAND "${CROSS_CXX}" ${KERNEL_DEFS} ${KERNEL_INCLUDES}
          -fPIC -O2 -std=c++17 -fvisibility=hidden -fstack-protector-all
          -c "${KERNEL_SRC}" -o "${KERNEL_OBJ}"
  COMMAND "${CROSS_CXX}" -shared "${KERNEL_OBJ}"
          -Wl,--whole-archive "${LIB64}/libaicpu_context.a" "${LIB64}/libbase_ascend_protobuf.a"
          -Wl,--no-whole-archive -Wl,-Bsymbolic
          -Wl,--exclude-libs=libbase_ascend_protobuf.a -s
          -o "${KERNEL_OUT}"
  DEPENDS "${KERNEL_SRC}" ${KERNEL_DEP_HEADERS}
  COMMENT "[kernel] 交叉编译 AICPU kernel ${KERNEL_SO}"
  VERBATIM)
add_custom_target(aicpu_kernel ALL DEPENDS "${KERNEL_OUT}")

# ==============================================================================
# 2. aclnn opapi（host 原生 CMake target）
# ==============================================================================
add_library(cust_opapi SHARED
  op_host/op_api/aclnn_${OP_NAME}.cpp
  op_host/op_api/l0_${OP_NAME}.cpp)

set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME cust_opapi PREFIX "lib")

target_compile_definitions(cust_opapi PRIVATE
  _GLIBCXX_USE_CXX11_ABI=0 google=ascend_private
  ACLNN_LOG_FMT_CHECK OP_LOG_LIBOPAPI_ONLY BUILD_OPEN_PROJECT BUILD_OPEN_PROJECT_API=1)

target_compile_options(cust_opapi PRIVATE
  -fPIC -O2 -fvisibility=hidden -fstack-protector-strong)

target_include_directories(cust_opapi PRIVATE
  ${ASCEND_HOME_PATH}/include
  ${ARCH_DIR}/include
  ${ARCH_DIR}/include/aclnn
  ${ASCEND_HOME_PATH}/include/aclnn
  ${ARCH_DIR}/pkg_inc/aicpu
  ${ARCH_DIR}/pkg_inc/aicpu_common/context
  ${CMAKE_CURRENT_SOURCE_DIR}/op_host
  ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api)
target_include_directories(cust_opapi SYSTEM PRIVATE
  ${ARCH_DIR}/include/aclnn
  ${ASCEND_HOME_PATH}/pkg_inc/base)

target_link_directories(cust_opapi PRIVATE ${LIB64})
# 显式确保不生成 soname（双重保险）
target_link_options(cust_opapi PRIVATE
  -s -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,-Bsymbolic)
target_link_libraries(cust_opapi PRIVATE
  ${LIB64}/libnnopbase.so
  profapi ge_common_base ascend_dump ascendalog dl
  ${LIB64}/libopapi_math.so)

# ==============================================================================
# 3. 打包 vendor 目录（cmake --build 时产出到 build/vendor/<NAME>）
# ==============================================================================
set(VENDOR_STAMP "${CMAKE_BINARY_DIR}/vendor/.stamp")
add_custom_command(
  OUTPUT "${VENDOR_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${VENDOR_STAGE}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${VENDOR_STAGE}/op_api/lib"
          "${VENDOR_STAGE}/op_api/include/aclnnop"
          "${VENDOR_STAGE}/op_impl/cpu/aicpu_kernel/impl"
          "${VENDOR_STAGE}/op_impl/cpu/config"
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:cust_opapi>" "${VENDOR_STAGE}/op_api/lib/"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api/aclnn_${OP_NAME}.h"
          "${VENDOR_STAGE}/op_api/include/aclnnop/"
  COMMAND ${CMAKE_COMMAND} -E copy "${KERNEL_OUT}" "${VENDOR_STAGE}/op_impl/cpu/aicpu_kernel/impl/"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/${OP_NAME}_aicpu.json"
          "${VENDOR_STAGE}/op_impl/cpu/config/cust_aicpu_kernel.json"
  COMMAND ${CMAKE_COMMAND} -E echo "Version=1.0" > "${VENDOR_STAGE}/version.info"
  COMMAND ${CMAKE_COMMAND} -E echo "load_priority=${VENDOR_NAME}" > "${CMAKE_BINARY_DIR}/vendor/config.ini"
  COMMAND ${CMAKE_COMMAND} -E touch "${VENDOR_STAMP}"
  DEPENDS cust_opapi "${KERNEL_OUT}"
          "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/${OP_NAME}_aicpu.json"
          "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api/aclnn_${OP_NAME}.h"
  COMMENT "[vendor] 打包 ${VENDOR_NAME}"
  VERBATIM)
add_custom_target(vendor ALL DEPENDS "${VENDOR_STAMP}")

# ==============================================================================
# 4. 安装到 opp/vendors（cmake --install）并合并 config.ini
# ==============================================================================
install(DIRECTORY "${VENDOR_STAGE}/" DESTINATION "${OPP_VENDORS}/${VENDOR_NAME}")
install(CODE "
  set(_cfg \"${OPP_VENDORS}/config.ini\")
  set(_name \"${VENDOR_NAME}\")
  if(EXISTS \"\${_cfg}\")
    file(READ \"\${_cfg}\" _content)
    if(NOT _content MATCHES \"\${_name}\")
      string(REGEX REPLACE \"load_priority=\" \"load_priority=\${_name},\" _content \"\${_content}\")
      file(WRITE \"\${_cfg}\" \"\${_content}\")
    endif()
  else()
    file(WRITE \"\${_cfg}\" \"load_priority=\${_name}\\n\")
  endif()
  message(STATUS \"[install] 完成。运行前请: export ASCEND_CUSTOM_OPP_PATH=${OPP_VENDORS}/\${_name}\")
")

# ==============================================================================
# 5. C++ 冒烟用例（可选：cmake --build build --target run_example）
# ==============================================================================
add_executable(test_aclnn_${OP_NAME} EXCLUDE_FROM_ALL examples/test_aclnn_${OP_NAME}.cpp)
add_dependencies(test_aclnn_${OP_NAME} vendor)
target_compile_options(test_aclnn_${OP_NAME} PRIVATE -fPIC)
target_include_directories(test_aclnn_${OP_NAME} PRIVATE
  ${ASCEND_HOME_PATH}/include
  ${ARCH_DIR}/include
  ${VENDOR_STAGE}/op_api/include
  ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu)
target_link_options(test_aclnn_${OP_NAME} PRIVATE
  -Wl,--allow-shlib-undefined -Wl,-rpath-link,${LIB64})
target_link_directories(test_aclnn_${OP_NAME} PRIVATE
  ${VENDOR_STAGE}/op_api/lib ${LIB64})
target_link_libraries(test_aclnn_${OP_NAME} PRIVATE cust_opapi ascendcl nnopbase c_sec)

add_custom_target(run_example
  COMMAND ${CMAKE_COMMAND} -E env
          ASCEND_CUSTOM_OPP_PATH=${VENDOR_STAGE}
          LD_LIBRARY_PATH=${VENDOR_STAGE}/op_api/lib:${LIB64}:$ENV{LD_LIBRARY_PATH}
          $<TARGET_FILE:test_aclnn_${OP_NAME}>
  DEPENDS test_aclnn_${OP_NAME}
  COMMENT "[example] 运行冒烟用例"
  VERBATIM USES_TERMINAL)
