cmake_minimum_required(VERSION 3.10)
 	 
project(keyed_jagged_index_select_dim1)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

add_library(keyed_jagged_index_select_dim1 SHARED keyed_jagged_index_select_dim1.cpp ${CMAKE_CURRENT_LIST_DIR}/../asynchronous_complete_cumsum/asynchronous_complete_cumsum.cpp)

target_compile_features(keyed_jagged_index_select_dim1 PRIVATE cxx_std_17)
target_compile_options(keyed_jagged_index_select_dim1 PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(keyed_jagged_index_select_dim1 PUBLIC c10 torch torch_cpu torch_npu c_sec)