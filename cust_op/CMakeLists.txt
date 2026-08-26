# ===========================================================================
# RecSDK cust_op: 算子构建 + 适配层编译 + whl 打包
# ===========================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(rec_cust_ops LANGUAGES CXX)

set(RECSDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Show compile commands" FORCE)

# ---------------------------------------------------------------------------
# Python 模块文件安装（rec_cust_ops/ 打入 whl 包根目录）
# ---------------------------------------------------------------------------
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rec_cust_ops/
    DESTINATION .
)

# ---------------------------------------------------------------------------
# Python / PyTorch / torch_npu 路径（优先使用 setup.py 传入的 CMAKE_PREFIX_PATH）
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 used by CMake: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")
message(STATUS "PyTorch prefix (from setup.py): ${CMAKE_PREFIX_PATH}")
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

# ABI：必须与当前 PyTorch 一致（1=C++11 std::string 新 ABI，0=旧 ABI）；符号不一致会导致 load_library 报 undefined symbol
# 优先使用 setup.py 通过 cmake_args 传入的 _GLIBCXX_USE_CXX11_ABI，以保证 CMake 与 setup.py 的 ABI 检测一致
if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; v=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 0); print(1 if v else 0)"
        OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
string(TOLOWER "${_GLIBCXX_USE_CXX11_ABI}" _GLIBCXX_USE_CXX11_ABI_LOWER)
if("${_GLIBCXX_USE_CXX11_ABI_LOWER}" STREQUAL "" OR "${_GLIBCXX_USE_CXX11_ABI_LOWER}" STREQUAL "true" OR "${_GLIBCXX_USE_CXX11_ABI_LOWER}" STREQUAL "false")
    set(_GLIBCXX_USE_CXX11_ABI 0)
endif()
if(NOT "${_GLIBCXX_USE_CXX11_ABI}" STREQUAL "1")
    set(_GLIBCXX_USE_CXX11_ABI 0)
endif()
message(STATUS "_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI} (must match torch lib for torch::Library symbol)")

# ---------------------------------------------------------------------------
# AscendC 算子编译与打包（逻辑在 RecOps.cmake，新增算子只需改该文件）
# ---------------------------------------------------------------------------
include(${RECSDK_SOURCE_DIR}/RecOps.cmake)

# ---------------------------------------------------------------------------
# 主库：torch_plugin 适配层（对齐 fbgemm 的 host library 模式）
# 针对 A5 / A3 / A2 分别构建适配层 .so，仅包含对应芯片的算子
# ---------------------------------------------------------------------------
function(recsdk_add_host_library target_name output_name npu_chip_flag)
    set(_adapter_srcs ${ARGN})
    add_library(${target_name} SHARED ${_adapter_srcs})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_compile_options(${target_name} PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
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
        # __init__.py 里会通过 import 加载 torch 库，无需编入 RPATH
    )
endfunction()

recsdk_add_host_library(rec_cust_ops_py_a5 rec_cust_ops_py_a5 1 ${RECSDK_ADAPTER_SRCS_A5})
recsdk_add_host_library(rec_cust_ops_py_a3 rec_cust_ops_py_a3 0 ${RECSDK_ADAPTER_SRCS_A3})
recsdk_add_host_library(rec_cust_ops_py_a2 rec_cust_ops_py_a2 0 ${RECSDK_ADAPTER_SRCS_A2})

# host library 依赖 AscendC 算子编译完成
foreach(_t ${ASCENDC_TARGETS})
    if(TARGET ${_t})
        add_dependencies(rec_cust_ops_py_a5 ${_t})
        add_dependencies(rec_cust_ops_py_a3 ${_t})
        add_dependencies(rec_cust_ops_py_a2 ${_t})
    endif()
endforeach()

# ---------------------------------------------------------------------------
# 安装：将 .so 安装到 Python 包目录
# ---------------------------------------------------------------------------
install(TARGETS rec_cust_ops_py_a5
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
install(TARGETS rec_cust_ops_py_a3
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
install(TARGETS rec_cust_ops_py_a2
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
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/rec_cust_ops/env_setup.sh
    DESTINATION .
)
