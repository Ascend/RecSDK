cmake_minimum_required(VERSION 3.10)

project(relative_attn_bias)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(relative_attn_bias SHARED relative_attn_bias_pos.cpp relative_attn_bias_time.cpp)

target_compile_features(relative_attn_bias PRIVATE cxx_std_17)
target_compile_options(relative_attn_bias PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(relative_attn_bias PUBLIC c10 torch torch_cpu torch_npu c_sec)
