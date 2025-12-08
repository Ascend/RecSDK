cmake_minimum_required(VERSION 3.10)

project(in_linear_silu)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(in_linear_silu SHARED in_linear_silu.cpp)

target_compile_features(in_linear_silu PRIVATE cxx_std_17)
target_compile_options(in_linear_silu PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(in_linear_silu PUBLIC c10 torch torch_cpu torch_npu c_sec)