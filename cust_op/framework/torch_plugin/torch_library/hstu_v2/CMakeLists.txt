cmake_minimum_required(VERSION 3.10)

# 生成 compile_commands.json 用于代码补全和静态分析
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(hstu_v2)

include("${CMAKE_CURRENT_LIST_DIR}/../common/CommonTorchOpConfig.cmake")

file(GLOB HSTU_OP_SRC
     "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
     "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

add_library(hstu_v2 SHARED ${HSTU_OP_SRC})

target_compile_features(hstu_v2 PRIVATE cxx_std_17)
target_compile_options(hstu_v2 PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

target_link_libraries(hstu_v2 PUBLIC c10 torch torch_cpu torch_npu c_sec)
