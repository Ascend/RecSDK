# 开启测试
enable_testing()
set(CMAKE_CXX_STANDARD 17)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_definitions(-DGTEST)

# src
file(GLOB_RECURSE MXREC_SRC ${PROJECT_SOURCE_DIR}/core/*.cpp)
message("MXREC_SRC: " ${MXREC_SRC})
file(GLOB_RECURSE MXREC_TEST_SRC ./*.cpp)
message("MXREC_TEST_SRC: " ${MXREC_TEST_SRC})

set(CMAKE_CXX_FLAGS "--coverage")

add_executable(test_main ${MXREC_SRC} ${MXREC_TEST_SRC})

if(NOT SECUREC_PATH)
    set(SECUREC_PATH ${PROJECT_SOURCE_DIR}/../platform/securec)
endif()

if (easy_profiler_FOUND)
    message("==link with easy_profiler==")
    target_link_directories(test_main PUBLIC ${EASY_PROFILER_PATH})
    target_link_libraries(test_main PUBLIC easy_profiler)
endif ()

include_directories(PRIVATE .)
target_include_directories(test_main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${ASCEND_DRIVER_PATH}/include
        ${SECUREC_PATH}/include
)

target_link_directories(test_main
    PRIVATE
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${SECUREC_PATH}/lib)

target_link_directories(test_main
        PUBLIC
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${HDF5_PATH}/lib
        ${SECUREC_PATH}/lib
        ${ASCEND_DRIVER_PATH}/lib64/driver
        )

target_link_libraries(test_main PUBLIC ${TF_LIB}
        securec
        OpenMP::OpenMP_CXX ${HDF5_CXX_LIBRARIES}  ${MPI_CXX_LIBRARIES}
        ${PYTHON_LIBRARY} drvdsmi_host
        -ldl
        )

target_link_libraries(test_main PUBLIC
        ${GTEST_BOTH_LIBRARIES}
        MPI::MPI_CXX)

target_link_libraries(test_main PUBLIC ascendcl msprofiler ge_executor gert runtime ge_common register graph ascend_protobuf
        profapi opt_feature error_manager exe_graph acl_tdt_channel acl_tdt_queue securec drvdsmi_host)