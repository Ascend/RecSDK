cmake_minimum_required(VERSION 3.10)

project(split_embedding_codegen_forward_unweighted)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

include_directories(${Python3_INCLUDE_DIRS})

file(GLOB_RECURSE ASCEND_LOAD_LIBRARY CONFIGURE_DEPENDS "*.cpp")
list(FILTER ASCEND_LOAD_LIBRARY EXCLUDE REGEX "(build|cmake)/.*")

add_library(split_embedding_codegen_forward_unweighted SHARED ${ASCEND_LOAD_LIBRARY})

target_compile_features(split_embedding_codegen_forward_unweighted PRIVATE cxx_std_17)
target_compile_options(split_embedding_codegen_forward_unweighted PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(split_embedding_codegen_forward_unweighted PUBLIC c10 torch torch_cpu torch_npu c_sec)