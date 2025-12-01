cmake_minimum_required(VERSION 3.10)

project(invert_permute)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(invert_permute SHARED invert_permute.cpp)

target_compile_features(invert_permute PRIVATE cxx_std_17)
target_compile_options(invert_permute PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(invert_permute PUBLIC c10 torch torch_cpu torch_npu c_sec)
