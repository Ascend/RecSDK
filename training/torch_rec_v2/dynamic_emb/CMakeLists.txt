cmake_minimum_required(VERSION 3.16)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
find_package(ASC REQUIRED)
project(dynamic_emb_extensions LANGUAGES ASC CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH}
        CACHE STRING "ASCEND CANN package installation directory"
    )
else()
    if(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
            CACHE STRING "ASCEND CANN package installation directory"
        )
    elseif(EXISTS "/usr/local/Ascend/latest")
        # CANN partial package installation
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/latest"
            CACHE STRING "ASCEND CANN package installation directory"
        )
    else()
        message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH is not specified")
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


# 收集AscendC自定义算子
file(GLOB_RECURSE KERNEL_FILES 
    "csrc/ops/*.cpp" 
    "csrc/ops/*.cc" 
    "csrc/ops/*.cxx"
    "csrc/ops/*.c"
)

message(STATUS "AscendC source files: ${KERNEL_FILES}")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

## ASC compile start
include(cmake/CommonTorchOpConfig.cmake)
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/dynamic_variable_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hkv_variable.cpp
    PROPERTIES LANGUAGE ASC
)

add_library(dynamic_variable_base SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/dynamic_variable_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hkv_variable.cpp
)

set_target_properties(dynamic_variable_base PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_include_directories(dynamic_variable_base PRIVATE
	${Python3_INCLUDE_DIRS}
	"${CMAKE_CURRENT_SOURCE_DIR}/csrc"
	"${CMAKE_CURRENT_SOURCE_DIR}/../../../cust_op/hkv/include"
)

target_compile_options(dynamic_variable_base PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-3510>
    -DUSE_RTTI
)

target_link_options(dynamic_variable_base PRIVATE -Wl,-z,relro,-z,now -s)

target_link_libraries(dynamic_variable_base PRIVATE
    ${Python3_LIBRARIES}
    m
    graph_base
    tiling_api
    register
    platform
    ascendalog
    dl
    opapi
    nnopbase
    "${PYTORCH_NPU_INSTALL_PATH}/lib/libtorch_npu.so"                                                                          
    "${PYTORCH_INSTALL_PATH}/lib/libc10.so"                                                                                    
    "${PYTORCH_INSTALL_PATH}/lib/libtorch.so"                                                                                  
    "${PYTORCH_INSTALL_PATH}/lib/libtorch_cpu.so" 
)

install(TARGETS dynamic_variable_base
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
## ASC compile end

# 收集主源码文件（排除ops目录）
file(GLOB_RECURSE MAIN_SOURCE_FILES 
    "csrc/*.cpp" 
    "csrc/*.cc" 
    "csrc/*.cxx"
    "csrc/*.c"
)

# 过滤掉ops目录和dynamic_variable_base_lib相关的文件
list(FILTER MAIN_SOURCE_FILES EXCLUDE REGEX ".*ops.*")
list(REMOVE_ITEM MAIN_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/csrc/dynamic_variable_base.cpp)
list(REMOVE_ITEM MAIN_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hkv_variable.cpp)
message(STATUS "Main source files: ${MAIN_SOURCE_FILES}")

# 构建主Python扩展
if(MAIN_SOURCE_FILES)
    message(STATUS "Building main Python extension...")
    include(cmake/CommonTorchOpConfig.cmake)

    # 创建Python扩展模块
    pybind11_add_module(${PROJECT_NAME} MODULE ${MAIN_SOURCE_FILES})

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${Python3_INCLUDE_DIRS}
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../cust_op/hkv/include"
        "${ASCEND_CANN_PACKAGE_PATH}/include"
    )

    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

    target_compile_options(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
        $<$<NOT:$<BOOL:${ENABLE_COVERAGE}>>: -O2> 
        -Wno-error=unused-variable
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fPIC
        -DUSE_RTTI
        $<$<BOOL:${ENABLE_COVERAGE}>: -O0 --coverage -g>
    )

    get_target_property(dep_options host_intf_pub INTERFACE_COMPILE_OPTIONS)
    if (dep_options)
        list(FILTER dep_options EXCLUDE REGEX ".*std=.*")
        list(FILTER dep_options EXCLUDE REGEX "-fvisibility-inlines-hidden")
        list(APPEND dep_options -std=c++17 -fvisibility-inlines-hidden)
        set_target_properties(host_intf_pub PROPERTIES INTERFACE_COMPILE_OPTIONS "${dep_options}")
    endif()

    # 添加链接器选项，确保库被包含
    target_link_options(${PROJECT_NAME} PRIVATE
        -fPIC -Wl,--no-as-needed  # 强制链接所有库，即使看似未使用
        $<$<BOOL:${ENABLE_COVERAGE}>:--coverage>
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${Python3_LIBRARIES}
        $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
        dynamic_emb_op_${RUN_MODE}
        tiling_api
        register
        platform
        ascendalog
        dl
        opapi
        nnopbase
        -Wl,--whole-archive  # 开始包含所有符号
        torch_npu
        -Wl,--no-whole-archive  # 结束
        c10
        torch
        torch_cpu
        c_sec
        dynamic_variable_base
    )
    # 添加链接选项
    target_link_options(${PROJECT_NAME} PRIVATE -fPIC)
    message(STATUS "✅ Main extension library: ${PROJECT_NAME}")
else()
    message(FATAL_ERROR "No main source files found")
endif()