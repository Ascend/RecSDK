cmake_minimum_required(VERSION 3.10)

project(offsets_range)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(offsets_range SHARED offsets_range.cpp)

target_compile_features(offsets_range PRIVATE cxx_std_17)
target_compile_options(offsets_range PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(offsets_range PUBLIC c10 torch torch_cpu torch_npu c_sec)
