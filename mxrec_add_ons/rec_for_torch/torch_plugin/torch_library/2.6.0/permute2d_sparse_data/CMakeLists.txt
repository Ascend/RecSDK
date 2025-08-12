cmake_minimum_required(VERSION 3.10)

project(permute2d_sparse_data)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(permute2d_sparse_data SHARED permute2d_sparse_data.cpp)

target_compile_features(permute2d_sparse_data PRIVATE cxx_std_17)
target_compile_options(permute2d_sparse_data PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(permute2d_sparse_data PUBLIC c10 torch torch_cpu torch_npu c_sec)