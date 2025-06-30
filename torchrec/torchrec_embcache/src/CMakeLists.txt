cmake_minimum_required(VERSION 3.10)
project(embedding_cache LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型默认值
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# 全局编译选项配置
set(COMMON_FLAGS
    -Wall
    -fPIC
    -fms-extensions
    -Wunused-variable
    -Wunused-value
    -Wcast-align
    -Wcast-qual
    -Winvalid-pch
    -Wwrite-strings
    -Wsign-compare
    -Wfloat-equal
    -Wextra
)

# 将列表转换为空格分隔的字符串
list(JOIN COMMON_FLAGS " " COMMON_FLAGS_STR)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "====== Build Type: Release")
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS_STR} -O3 -D_FORTIFY_SOURCE=2 -fstack-protector-all -fstack-protector-strong")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "====== Build Type: Debug")
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS_STR} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "UT")
    message(STATUS "====== Build Type: UT")
    set(CXX_FLAGS "${COMMON_FLAGS_STR} -g -fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -fstack-protector-all")
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()

# 安装路径
if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/cmake_build/install")
    message(STATUS "CMAKE_INSTALL_PREFIX set to default: ${CMAKE_INSTALL_PREFIX}")
endif()

# 三方库配置
add_subdirectory(3rdparty)

# for securec
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/securec/include)
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/securec/lib)

# 主项目配置
add_subdirectory(torchrec_embcache)