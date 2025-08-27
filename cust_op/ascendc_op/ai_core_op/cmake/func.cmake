set(CMAKE_FUNC_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC}                   
                  -D_GLIBCXX_USE_CXX11_ABI=0 -DLOG_CPP
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include
                  -I ${CMAKE_FUNC_DIR}/../common
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()

  set(ENV{ENABLE_SOURCE_PACAKGE} ${OPBUILD_ENABLE_SOURCE})
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  unset(ENV{ENABLE_SOURCE_PACAKGE})
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

include_directories(${CMAKE_FUNC_DIR}/../common)