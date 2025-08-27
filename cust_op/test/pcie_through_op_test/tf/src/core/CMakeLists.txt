cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

if(NOT ABSEIL_PATH)
    set(ABSEIL_PATH ${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow_core/)
endif()

if(NOT SECUREC_PATH)
    set(SECUREC_PATH ${PROJECT_SOURCE_DIR}/../platform/securec)
endif()

include_directories(${ABSEIL_PATH}/include)
link_directories(${ABSEIL_PATH}/lib)


if (${TF_PATH} MATCHES "tensorflow_core")
    include_directories(${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow_core/include)
else()
    include_directories(${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow/include)
endif()

link_libraries(stdc++fs)

file(GLOB_RECURSE HOST_SRC ./*.cpp ./*.h)
add_library(SRC SHARED ${HOST_SRC}
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/common_func/common_func.cpp
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/pcie_through/rma_shm_svm.cpp
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/pcie_through/
        ${PROJECT_SOURCE_DIR}/../../../../../training/tf_rec_v1/src/core/utils/common.cpp
        ${PROJECT_SOURCE_DIR}/../../../../../training/tf_rec_v1/src/core/utils/common.h
        ${PROJECT_SOURCE_DIR}/../../../../../training/tf_rec_v1/src/core/utils/config.cpp
        ${PROJECT_SOURCE_DIR}/../../../../../training/tf_rec_v1/src/core/utils/config.h
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/error/error.cpp
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/error/error.h
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/log/logger.cpp
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core/log/logger.h)

target_include_directories(SRC
        PUBLIC
        ${SECUREC_PATH}/include
        ${ASCEND_DRIVER_PATH}/include
        ${ASCEND_DRIVER_PATH}/kernel/inc
        ${PROJECT_SOURCE_DIR}/../../../../../training/common/src/core
        ${PROJECT_SOURCE_DIR}/../../../../../training/tf_rec_v1/src/core
        )

target_link_directories(SRC
        PUBLIC
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${SECUREC_PATH}/lib
        )

target_link_libraries(SRC PUBLIC
        ascendcl runtime drvdsmi_host ascend_hal securec
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES}
        ${PYTHON_LIBRARY}
        -l:_tf_adapter.so
        )

install(TARGETS SRC LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})