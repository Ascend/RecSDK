cmake_minimum_required(VERSION 3.10)

project(ln_mul)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(ln_mul SHARED ln_mul.cpp)

target_compile_features(ln_mul PRIVATE cxx_std_17)
target_compile_options(ln_mul PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(ln_mul PUBLIC c10 torch torch_cpu torch_npu c_sec)