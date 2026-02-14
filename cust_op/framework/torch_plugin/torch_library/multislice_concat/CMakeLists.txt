cmake_minimum_required(VERSION 3.10)

project(multislice_concat)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(multislice_concat SHARED multislice_concat.cpp)

target_compile_features(multislice_concat PRIVATE cxx_std_17)
target_compile_options(multislice_concat PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(multislice_concat PUBLIC c10 torch torch_cpu torch_npu c_sec)
