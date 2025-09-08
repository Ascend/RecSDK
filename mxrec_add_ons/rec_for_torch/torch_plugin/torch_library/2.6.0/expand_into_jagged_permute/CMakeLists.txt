cmake_minimum_required(VERSION 3.10)

project(expand_into_jagged_permute)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(expand_into_jagged_permute SHARED expand_into_jagged_permute.cpp)

target_compile_features(expand_into_jagged_permute PRIVATE cxx_std_17)
target_compile_options(expand_into_jagged_permute PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(expand_into_jagged_permute PUBLIC c10 torch torch_cpu torch_npu c_sec)