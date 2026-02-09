cmake_minimum_required(VERSION 3.10)

project(segment_sum_csr)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(segment_sum_csr SHARED segment_sum_csr.cpp)

target_compile_features(segment_sum_csr PRIVATE cxx_std_17)
target_compile_options(segment_sum_csr PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(segment_sum_csr PUBLIC c10 torch torch_cpu torch_npu c_sec)