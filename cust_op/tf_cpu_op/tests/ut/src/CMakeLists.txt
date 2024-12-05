add_compile_options(-ftest-coverage -fprofile-arcs)
link_libraries(gcov)

set(REC_BASE_UTIL_INSTALL_DIR       ${PROJECT_SOURCE_DIR}/install)
set(REC_BASE_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
message("src" ${OCK_RECBASE_SRC_DIR})

# 包含所有组件的cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/../conf/toolchain.cmake")
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

file(GLOB_RECURSE TEST_REC_BASE_FILES *.cpp *.h)
add_executable(test_rec_base ${TEST_REC_BASE_FILES})
include_directories(${REC_BASE_UTIL_INSTALL_DIR}/googletest/include)
link_directories(${REC_BASE_UTIL_INSTALL_DIR}/googletest/lib64)

SET(LIB_3RD_GMOCK ${REC_BASE_UTIL_INSTALL_DIR}/googletest/lib64/libgmock.a)
SET(LIB_3RD_GTEST ${REC_BASE_UTIL_INSTALL_DIR}/googletest/lib64/libgtest.a)

message(${REC_BASE_SRC_DIR}/include)
# 添加库文件的搜索路径
target_link_directories(test_rec_base
            PUBLIC
            ${PROJECT_SOURCE_DIR}/output/rec_base/lib
            )

# 添加头文件的搜索路径
target_include_directories(test_rec_base
        PUBLIC
        ${REC_BASE_SRC_DIR}/include
        ${PROJECT_SOURCE_DIR}
        ${REC_BASE_SRC_DIR}/common/util
        )

# 用来指定要链接的库
target_link_libraries(test_rec_base
        PUBLIC
        -Wl,--start-group
        _rec_base
        pthread
        dl
        ${LIB_3RD_GTEST}
        ${LIB_3RD_GMOCK}
        -Wl,--end-group)

# 打印构建选项
get_target_property(COMPILE_FLAGS test_rec_base COMPILE_OPTIONS)
get_target_property(LINK_FLAGS test_rec_base LINK_OPTIONS)
message(STATUS "cOMPILER ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compile flags: ${COMPILE_FLAGS}")
message(STATUS "Link flags: ${LINK_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")