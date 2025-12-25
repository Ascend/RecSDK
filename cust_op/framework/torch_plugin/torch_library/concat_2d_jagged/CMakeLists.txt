cmake_minimum_required(VERSION 3.10)

project(concat_2d_jagged)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(concat_2d_jagged SHARED concat_jagged_tensor.cpp)

target_compile_features(concat_2d_jagged PRIVATE cxx_std_17)
target_compile_options(concat_2d_jagged PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(concat_2d_jagged PUBLIC c10 torch torch_cpu torch_npu c_sec)
