cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kvcache_npu_cpp LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# SOC_VERSION
if(DEFINED ENV{SOC_VERSION})
    set(SOC_VERSION $ENV{SOC_VERSION})
else()
    set(SOC_VERSION "Ascend950PR_9579")
endif()

# ASCEND_CANN_PACKAGE_PATH
if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})
elseif(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
else()
    message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH not set and no default found")
endif()

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# Include common torch/torch_npu config
include(cmake/CommonTorchOpConfig.cmake)

# Define AscendC kernel source files
set(KERNEL_FILES
    ${PROJECT_ROOT}/src/kernels/scatter_paged_kvcache/scatter_paged_kvcache.cpp
    ${PROJECT_ROOT}/src/kernels/gather_paged_kvcache/gather_paged_kvcache.cpp
    ${PROJECT_ROOT}/src/kernels/get_paged_batch_indices_positions/get_paged_batch_indices_positions.cpp
    ${PROJECT_ROOT}/src/kernels/append_paged_kvcache/append_paged_kvcache.cpp
)

# Include directories for kernel compilation
include_directories(
    ${PROJECT_ROOT}/src/kernels
)

# Step 1: Compile AscendC device kernels
include(cmake/npu_kernel.cmake)

# Ensure kernel library outputs to same directory as Python extension
if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(kvcache_kernel PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

# Step 2: Compile host Python extension
# After ascendc_library(), CMAKE_CURRENT_BINARY_DIR contains aclrtlaunch_*.h
pybind11_add_module(kvcache_npu_cpp MODULE
    src/pybind.cpp
    src/native_host_kvcache_manager_impl.cpp
    src/npu_kvcache_manager_impl.cpp
    src/gather_scatter.cpp
)

target_include_directories(kvcache_npu_cpp PRIVATE
    ${PROJECT_ROOT}/src
    ${PROJECT_ROOT}/src/kernels
    ${CMAKE_CURRENT_BINARY_DIR}  # aclrtlaunch_*.h lives here
)

target_compile_options(kvcache_npu_cpp PRIVATE
    -O3 -fvisibility=hidden -DWITH_PYBIND11=1
)

target_link_libraries(kvcache_npu_cpp PRIVATE
    ascendcl
    kvcache_kernel
    torch
    torch_cpu
    c10
    torch_npu
)

# Set RPATH to find libscatter_kvcache_kernel.so at runtime
set_target_properties(kvcache_npu_cpp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
)
