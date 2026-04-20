cmake_minimum_required(VERSION 3.10)

project(lengths_index)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(lengths_index SHARED
        lengths_index.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../asynchronous_complete_cumsum/asynchronous_complete_cumsum.cpp
)

target_compile_features(lengths_index PRIVATE cxx_std_17)
target_compile_options(lengths_index PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(lengths_index PUBLIC c10 torch torch_cpu torch_npu c_sec)