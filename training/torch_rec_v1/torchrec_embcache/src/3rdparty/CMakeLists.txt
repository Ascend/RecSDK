cmake_minimum_required(VERSION 3.10)

# SecureC 配置（非CMake项目）
set(SECUREC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/securec)
file(GLOB SECUREC_SOURCES
    ${SECUREC_DIR}/src/*.c
    ${SECUREC_DIR}/src/*.cpp
)

add_library(securec STATIC ${SECUREC_SOURCES})
target_include_directories(securec PUBLIC
    ${SECUREC_DIR}/include
)
target_compile_options(securec PRIVATE
    -Wall -Werror -fPIC
)