cmake_minimum_required(VERSION 3.10)

project(norm_multiply_dropout)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(norm_multiply_dropout SHARED norm_multiply_dropout.cpp)

target_compile_features(norm_multiply_dropout PRIVATE cxx_std_17)
target_compile_options(norm_multiply_dropout PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(norm_multiply_dropout PUBLIC c10 torch torch_cpu torch_npu c_sec)
