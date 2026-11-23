cmake_minimum_required(VERSION 3.10)

project(block_bucketize_sparse_features)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(
    block_bucketize_sparse_features
    SHARED
    block_bucketize_sparse_features.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../asynchronous_complete_cumsum/asynchronous_complete_cumsum.cpp
)

target_compile_features(block_bucketize_sparse_features PRIVATE cxx_std_17)
target_compile_options(block_bucketize_sparse_features PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(block_bucketize_sparse_features PUBLIC c10 torch torch_cpu torch_npu c_sec)
