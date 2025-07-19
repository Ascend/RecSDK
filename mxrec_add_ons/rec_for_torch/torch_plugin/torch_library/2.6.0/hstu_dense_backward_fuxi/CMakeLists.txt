cmake_minimum_required(VERSION 3.10)

project(hstu_dense_fuxi_backward_ops)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(hstu_dense_fuxi_backward_ops SHARED HstuDenseNpuFusionFuxi.cpp)

target_compile_features(hstu_dense_fuxi_backward_ops PRIVATE cxx_std_17)
target_compile_options(hstu_dense_fuxi_backward_ops PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(hstu_dense_fuxi_backward_ops PUBLIC c10 torch torch_cpu torch_npu c_sec)
