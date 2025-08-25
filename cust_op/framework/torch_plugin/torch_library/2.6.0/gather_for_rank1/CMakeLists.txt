cmake_minimum_required(VERSION 3.10)

project(gather_for_rank1)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(gather_for_rank1 SHARED gather_for_rank1.cpp)

target_compile_features(gather_for_rank1 PRIVATE cxx_std_17)
target_compile_options(gather_for_rank1 PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(gather_for_rank1 PUBLIC c10 torch torch_cpu torch_npu c_sec)
