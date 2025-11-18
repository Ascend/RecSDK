cmake_minimum_required(VERSION 3.10)

project(hstu_dense_ops)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

file(GLOB_RECURSE ASCEND_LOAD_LIBRARY CONFIGURE_DEPENDS "*.cpp")
list(FILTER ASCEND_LOAD_LIBRARY EXCLUDE REGEX "(build|cmake)/.*")

add_library(hstu_dense_ops SHARED ${ASCEND_LOAD_LIBRARY})

target_compile_features(hstu_dense_ops PRIVATE cxx_std_17)
target_compile_options(hstu_dense_ops PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(hstu_dense_ops PUBLIC c10 torch torch_cpu torch_npu c_sec)
