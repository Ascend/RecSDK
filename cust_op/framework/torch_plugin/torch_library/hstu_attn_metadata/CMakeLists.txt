cmake_minimum_required(VERSION 3.10)

project(hstu_attn_metadata)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(hstu_attn_metadata SHARED hstu_attn_metadata.cpp)

target_compile_features(hstu_attn_metadata PRIVATE cxx_std_17)
target_compile_options(hstu_attn_metadata PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(hstu_attn_metadata PUBLIC c10 torch torch_cpu torch_npu c_sec)
