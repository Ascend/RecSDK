cmake_minimum_required(VERSION 3.10)

project(gen_position_ids_with_timestamp)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(gen_position_ids_with_timestamp SHARED gen_position_ids_with_timestamp.cpp)

target_compile_features(gen_position_ids_with_timestamp PRIVATE cxx_std_17)
target_compile_options(gen_position_ids_with_timestamp PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(gen_position_ids_with_timestamp PUBLIC c10 torch torch_cpu torch_npu c_sec)