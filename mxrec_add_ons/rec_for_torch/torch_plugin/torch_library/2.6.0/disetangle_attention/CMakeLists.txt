cmake_minimum_required(VERSION 3.10)

project(disentangle_attention)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(disentangle_attention SHARED DisetangleAttenFusion.cpp)

target_compile_features(disentangle_attention PRIVATE cxx_std_17)
target_compile_options(disentangle_attention PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(disentangle_attention PUBLIC c10 torch torch_cpu torch_npu c_sec)
