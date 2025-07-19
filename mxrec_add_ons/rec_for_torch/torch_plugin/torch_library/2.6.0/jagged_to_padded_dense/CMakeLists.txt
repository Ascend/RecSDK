cmake_minimum_required(VERSION 3.10)

project(jagged_to_padded_dense)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(jagged_to_padded_dense SHARED jagged_to_padded_dense.cpp)

target_compile_features(jagged_to_padded_dense PRIVATE cxx_std_17)
target_compile_options(jagged_to_padded_dense PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(jagged_to_padded_dense PUBLIC c10 torch torch_cpu torch_npu c_sec)