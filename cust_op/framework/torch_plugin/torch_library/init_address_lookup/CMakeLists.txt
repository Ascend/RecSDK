cmake_minimum_required(VERSION 3.10)

project(init_address_lookup)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(init_address_lookup SHARED init_address_lookup.cpp)

target_compile_features(init_address_lookup PRIVATE cxx_std_17)
target_compile_options(init_address_lookup PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(init_address_lookup PUBLIC c10 torch torch_cpu torch_npu c_sec)
