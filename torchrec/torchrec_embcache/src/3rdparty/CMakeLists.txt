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

# glog配置（带编译选项）
set(BUILD_TESTING OFF CACHE BOOL "Disable glog tests")
set(WITH_GFLAGS OFF CACHE BOOL "Disable gflags")
set(WITH_UNWIND OFF CACHE BOOL "Disable libunwind")
add_subdirectory(glog)

set(THIRD_PARTY_TARGETS
    securec
    glog
    CACHE INTERNAL "Third party targets")

if (${CMAKE_BUILD_TYPE} MATCHES "UT|Debug")
    # googletest配置（禁用GoogleTest自身的测试）
    set(BUILD_GMOCK ON CACHE BOOL "Build Google Mock")
    set(INSTALL_GTEST OFF CACHE BOOL "Disable installation")
    add_subdirectory(googletest)

    set(THIRD_PARTY_TARGETS
        ${THIRD_PARTY_TARGETS}  # 保留已有目标
        gtest
        gtest_main
        gmock
        CACHE INTERNAL "Third party targets" FORCE  # 强制更新缓存
    )
endif ()