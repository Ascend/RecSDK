cmake_minimum_required(VERSION 3.10)

project(asynchronous_complete_cumsum)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(asynchronous_complete_cumsum SHARED asynchronous_complete_cumsum.cpp)

target_compile_features(asynchronous_complete_cumsum PRIVATE cxx_std_17)
target_compile_options(asynchronous_complete_cumsum PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(asynchronous_complete_cumsum PUBLIC c10 torch torch_cpu torch_npu c_sec)
