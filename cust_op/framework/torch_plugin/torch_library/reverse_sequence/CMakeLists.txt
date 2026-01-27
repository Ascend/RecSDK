cmake_minimum_required(VERSION 3.10)

project(reverse_sequence)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(reverse_sequence SHARED reverse_sequence.cpp)

target_compile_features(reverse_sequence PRIVATE cxx_std_17)
target_compile_options(reverse_sequence PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(reverse_sequence PUBLIC c10 torch torch_cpu torch_npu c_sec)
