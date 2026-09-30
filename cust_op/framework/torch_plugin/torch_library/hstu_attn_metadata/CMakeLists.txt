cmake_minimum_required(VERSION 3.10)

project(hstu_attn_metadata)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

# 前向 + 反向绑定编进同一个 .so：两边都是 TORCH_LIBRARY_FRAGMENT(mxrec)，共存无冲突，
# 与 common/ 聚合构建（GLOB 全部 .cpp 进 libfbgemm_npu_api.so）行为一致。
add_library(hstu_attn_metadata SHARED hstu_attn_metadata.cpp hstu_attn_metadata_backward.cpp)

target_compile_features(hstu_attn_metadata PRIVATE cxx_std_17)
target_compile_options(hstu_attn_metadata PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(hstu_attn_metadata PUBLIC c10 torch torch_cpu torch_npu c_sec)
