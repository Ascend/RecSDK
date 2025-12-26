cmake_minimum_required(VERSION 3.10)

project(permute_pooled_embs)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(permute_pooled_embs SHARED permute_pooled_embs.cpp)

target_compile_features(permute_pooled_embs PRIVATE cxx_std_17)

target_compile_options(permute_pooled_embs PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(permute_pooled_embs PUBLIC c10 torch torch_cpu torch_npu c_sec)
