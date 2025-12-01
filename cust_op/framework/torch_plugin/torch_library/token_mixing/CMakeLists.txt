cmake_minimum_required(VERSION 3.10)

project(token_mixing)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(token_mixing SHARED token_mixing.cpp)

target_compile_features(token_mixing PRIVATE cxx_std_17)
target_compile_options(token_mixing PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(token_mixing PUBLIC c10 torch torch_cpu torch_npu c_sec)