message("build mode " ${BUILD_MODE})

set(PLATFORM_UTILITIES_3RDPARTY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../../opensource)
set(PLATFORM_UTILITIES_3RDPARTY_BUILD_DIR ${PROJECT_SOURCE_DIR}/../../../opensource)
set(PLATFORM_UTILITIES_3RDPARTY_INSTALL_DIR ${PROJECT_SOURCE_DIR}/install)
set(GTEST_SOURCE_DIR ${PLATFORM_UTILITIES_3RDPARTY_SOURCE_DIR}/googletest-release-1.8.1)
set(GTEST_BUILD_DIR ${PLATFORM_UTILITIES_3RDPARTY_BUILD_DIR}/googletest-release-1.8.1)
set(GTEST_INSTALL_DIR ${PLATFORM_UTILITIES_3RDPARTY_INSTALL_DIR}/googletest-release-1.8.1)
set(SECUREC_SOURCE_DIR ${PLATFORM_UTILITIES_3RDPARTY_SOURCE_DIR}/securec)
set(SECUREC_BUILD_DIR ${PLATFORM_UTILITIES_3RDPARTY_BUILD_DIR}/securec)
set(SECUREC_INSTALL_DIR ${PLATFORM_UTILITIES_3RDPARTY_INSTALL_DIR}/securec)

add_definitions(_DDOFUN)
set(DOFUN "FALSE")

if (${BUILD_MODE} MATCHES "ut")
    set(DOFUN "TRUE")
endif (${BUILD_MODE} MATCHES "ut")

message("build securec")
# create build dir
exec_program(mkdir ${PLATFORM_UTILITIES_3RDPARTY_BUILD_DIR} ARGS -p ${SECUREC_BUILD_DIR})
exec_program(mkdir ${PLATFORM_UTILITIES_3RDPARTY_INSTALL_DIR} ARGS -p ${SECUREC_INSTALL_DIR})

# execute make && make install
exec_program(make ${SECUREC_SOURCE_DIR}/src ARGS -j)

#    scp -r ${SECUREC_SRC_PATH}/../include ${SECUREC_INSTALL_PATH}/
#    scp -r ${SECUREC_SRC_PATH}/../lib ${SECUREC_INSTALL_PATH}/
exec_program(scp ARGS -r ${SECUREC_SOURCE_DIR}/include ${SECUREC_INSTALL_DIR})
exec_program(scp ARGS -r ${SECUREC_SOURCE_DIR}/lib ${SECUREC_INSTALL_DIR})


message(============ ${DOFUN})
if (${DOFUN} MATCHES TRUE)
    message("build gTest")
    # create build dir
    exec_program(mkdir ${PLATFORM_UTILITIES_3RDPARTY_BUILD_DIR} ARGS -p ${GTEST_BUILD_DIR})

    # configure
    exec_program(cmake ${GTEST_BUILD_DIR} ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR} -DCMAKE_INSTALL_LIBDIR=lib64 ${GTEST_SOURCE_DIR})

    # execute make && make install
    exec_program(make ${GTEST_BUILD_DIR} ARGS clean)
    exec_program(make ${GTEST_BUILD_DIR} ARGS -j8)
    exec_program("make install" ${GTEST_BUILD_DIR})
endif ()