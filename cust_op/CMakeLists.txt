# ===========================================================================
# RecSDK cust_op: 算子构建 + 适配层编译 + whl 打包
# ===========================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(rec_ops LANGUAGES CXX)

set(RECSDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Show compile commands" FORCE)

# ---------------------------------------------------------------------------
# Python 模块文件安装（rec_sdk_ops/ 打入 whl 包根目录）
# ---------------------------------------------------------------------------
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rec_sdk_ops/
    DESTINATION .
)

# ---------------------------------------------------------------------------
# Python / PyTorch / torch_npu 路径（优先使用 setup.py 传入的 CMAKE_PREFIX_PATH）
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(CMAKE_PREFIX_PATH)
    set(PYTORCH_INSTALL_PATH "${CMAKE_PREFIX_PATH}")
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTORCH_INSTALL_PATH "${PYTHON_SITE_PACKAGES}/torch")
endif()
get_filename_component(PYTHON_SITE_PACKAGES "${PYTORCH_INSTALL_PATH}/.." ABSOLUTE)
set(PYTORCH_NPU_INSTALL_PATH "${PYTHON_SITE_PACKAGES}/torch_npu")
if(NOT EXISTS "${PYTORCH_NPU_INSTALL_PATH}")
    message(WARNING "torch_npu not found at ${PYTORCH_NPU_INSTALL_PATH}; NPU build may fail.")
endif()

# CANN 系统路径：securec.h 等
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
else()
    set(ASCEND_CANN_PACKAGE_PATH "")
endif()
if(ASCEND_CANN_PACKAGE_PATH)
    set(CANN_INCLUDE_DIR "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    if(IS_DIRECTORY "${CANN_INCLUDE_DIR}")
        message(STATUS "CANN include (securec.h etc.): ${CANN_INCLUDE_DIR}")
    else()
        message(WARNING "CANN include dir not found: ${CANN_INCLUDE_DIR}")
        set(CANN_INCLUDE_DIR "")
    endif()
else()
    set(CANN_INCLUDE_DIR "")
    message(STATUS "ASCEND_CANN_PACKAGE_PATH not set; securec.h may be missing.")
endif()

# CANN driver
set(ASCEND_DRIVER_PATH "$ENV{ASCEND_DRIVER_PATH}")
if(NOT ASCEND_DRIVER_PATH)
    set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver")
endif()
if(IS_DIRECTORY "${ASCEND_DRIVER_PATH}/kernel/libc_sec/include")
    set(DRIVER_LIBC_SEC_INCLUDE "${ASCEND_DRIVER_PATH}/kernel/libc_sec/include")
else()
    set(DRIVER_LIBC_SEC_INCLUDE "")
endif()

# ABI：必须与当前 PyTorch 一致
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; v=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 0); print(1 if v else 0)"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if("${TORCH_CXX11_ABI}" STREQUAL "" OR "${TORCH_CXX11_ABI}" STREQUAL "True" OR "${TORCH_CXX11_ABI}" STREQUAL "False")
    set(TORCH_CXX11_ABI 0)
endif()
if(NOT "${TORCH_CXX11_ABI}" STREQUAL "1")
    set(TORCH_CXX11_ABI 0)
endif()
message(STATUS "TORCH_CXX11_ABI=${TORCH_CXX11_ABI}")

# ---------------------------------------------------------------------------
# AscendC 算子编译与打包（逻辑在 RecOps.cmake，新增算子只需改该文件）
# ---------------------------------------------------------------------------
include(${RECSDK_SOURCE_DIR}/RecOps.cmake)

# ---------------------------------------------------------------------------
# 主库：torch_plugin 适配层（对齐 fbgemm 的 host library 模式）
# 针对 A5 (c310) 与 A2/A3 (v220) 构建两份适配层 .so
# ---------------------------------------------------------------------------
function(recsdk_add_host_library target_name output_name npu_chip_flag)
    add_library(${target_name} SHARED ${RECSDK_ADAPTER_SRCS})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
        -DNPU_CHIP_A5=${npu_chip_flag}
    )

    set(RECSDK_EXTRA_INCLUDES "")
    if(CANN_INCLUDE_DIR)
        list(APPEND RECSDK_EXTRA_INCLUDES ${CANN_INCLUDE_DIR})
    endif()
    if(DRIVER_LIBC_SEC_INCLUDE)
        list(APPEND RECSDK_EXTRA_INCLUDES ${DRIVER_LIBC_SEC_INCLUDE})
    endif()
    target_include_directories(${target_name} PRIVATE
        ${RECSDK_SOURCE_DIR}/framework/torch_plugin/torch_library/common
        ${PYTORCH_NPU_INSTALL_PATH}/include
        ${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc
        ${PYTORCH_INSTALL_PATH}/include
        ${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
        ${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed
        ${Python3_INCLUDE_DIRS}
        ${RECSDK_EXTRA_INCLUDES}
    )

    target_link_directories(${target_name} PRIVATE
        ${PYTORCH_INSTALL_PATH}/lib
        ${PYTORCH_NPU_INSTALL_PATH}/lib
        ${ASCEND_DRIVER_PATH}/lib64/common
    )

    set(RECSDK_TORCH_LIBS c10 torch torch_cpu torch_npu)
    find_library(TORCH_PYTHON_LIB torch_python HINTS "${PYTORCH_INSTALL_PATH}/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIB)
        list(APPEND RECSDK_TORCH_LIBS torch_python)
        message(STATUS "Linking ${target_name} with torch_python: ${TORCH_PYTHON_LIB}")
    endif()
    # c_sec 链接（与各算子独立 CMakeLists.txt 中的 c_sec 对应）
    find_library(C_SEC_LIB c_sec HINTS "${ASCEND_DRIVER_PATH}/lib64/common" NO_DEFAULT_PATH)
    if(C_SEC_LIB)
        list(APPEND RECSDK_TORCH_LIBS c_sec)
    endif()
    target_link_libraries(${target_name} PRIVATE ${RECSDK_TORCH_LIBS} Python3::Python)

    set_target_properties(${target_name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${output_name}
        BUILD_RPATH "${PYTORCH_INSTALL_PATH}/lib;${PYTORCH_NPU_INSTALL_PATH}/lib"
        INSTALL_RPATH "${PYTORCH_INSTALL_PATH}/lib;${PYTORCH_NPU_INSTALL_PATH}/lib"
    )
endfunction()

# 始终构建 A5 和 A2A3 两份 host library，运行时按 SOC 检测选择加载
recsdk_add_host_library(rec_sdk_ops_py_a5 rec_sdk_ops_py_a5 1)
recsdk_add_host_library(rec_sdk_ops_py_a2a3 rec_sdk_ops_py_a2a3 0)

# host library 依赖 AscendC 算子编译完成
foreach(_t ${ASCENDC_TARGETS})
    if(TARGET ${_t})
        add_dependencies(rec_sdk_ops_py_a5 ${_t})
        add_dependencies(rec_sdk_ops_py_a2a3 ${_t})
    endif()
endforeach()

# ---------------------------------------------------------------------------
# 安装：将 .so 安装到 Python 包目录
# ---------------------------------------------------------------------------
install(TARGETS rec_sdk_ops_py_a5
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
install(TARGETS rec_sdk_ops_py_a2a3
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# 将 staging 目录中的 AscendC 算子产物打入 whl 包
if(ASCENDC_STAGE_SUBDIRS)
    foreach(_opp_ver ${ASCENDC_STAGE_SUBDIRS})
        install(DIRECTORY ${ASCENDC_STAGE_ROOT}/${_opp_ver}/
            DESTINATION custom_opp/${_opp_ver}
            OPTIONAL)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# 将 env_setup.sh 打入包目录，供用户 source
# ---------------------------------------------------------------------------
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/rec_sdk_ops/env_setup.sh
    DESTINATION .
)
