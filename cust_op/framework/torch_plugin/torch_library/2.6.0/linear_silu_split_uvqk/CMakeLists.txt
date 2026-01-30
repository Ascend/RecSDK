cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(linear_silu_split_uvqk)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

find_package(Python3 COMPONENTS Interpreter Development)

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")

find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

link_directories(${ASCEND_HOME_PATH}/lib64
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib)

file(GLOB TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(linear_silu_split_uvqk_torch
        SHARED ${TORCH_BINDINGS_SRC})
target_compile_features(linear_silu_split_uvqk_torch PRIVATE cxx_std_17)
target_compile_options(linear_silu_split_uvqk_torch PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})
target_link_libraries(linear_silu_split_uvqk_torch PUBLIC c10 torch torch_cpu torch_npu c_sec)
install(TARGETS linear_silu_split_uvqk_torch DESTINATION . COMPONENT linear_silu_split_uvqk_torch)
