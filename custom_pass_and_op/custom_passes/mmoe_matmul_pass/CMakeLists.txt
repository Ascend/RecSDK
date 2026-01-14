cmake_minimum_required(VERSION 3.5.1)
project(mmoe_matmul_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

message(ASCEND_PATH)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
    -std=c++11
    -Wall
    -g
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

############ libfuse_tile_concat_pass.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES output_name ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${ASCEND_PATH}/include/graph
    ${ASCEND_PATH}/include/ge
    ${ASCEND_PATH}/include/register
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/x86_64-linux/include
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/compiler/lib64
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)
