cmake_minimum_required(VERSION 3.10)

project(fbgemm_npu_api)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/CommonTorchOpConfig.cmake")

include_directories(${Python3_INCLUDE_DIRS})

file(GLOB_RECURSE ASCEND_LOAD_LIBRARY CONFIGURE_DEPENDS "../*.cpp")
list(FILTER ASCEND_LOAD_LIBRARY EXCLUDE REGEX ".*/(build|cmake)/.*")

add_library(fbgemm_npu_api SHARED ${ASCEND_LOAD_LIBRARY})

target_compile_features(fbgemm_npu_api PRIVATE cxx_std_17)
target_compile_options(fbgemm_npu_api PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(fbgemm_npu_api PUBLIC c10 torch torch_cpu torch_npu c_sec)
