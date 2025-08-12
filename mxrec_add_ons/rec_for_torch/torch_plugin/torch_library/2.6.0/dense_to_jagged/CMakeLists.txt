cmake_minimum_required(VERSION 3.10)

project(dense_to_jagged)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(dense_to_jagged SHARED dense_to_jagged.cpp)

target_compile_features(dense_to_jagged PRIVATE cxx_std_17)
target_compile_options(dense_to_jagged PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(dense_to_jagged PUBLIC c10 torch torch_cpu torch_npu c_sec)
