cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

file(GLOB_RECURSE MXREC_OP_EOS_DATA ./*.cc)
add_library(rec_eos_ops SHARED ${MXREC_OP_EOS_DATA})

set(TF_INTERNAL_LIB ${TF_PATH}/python/_pywrap_tensorflow_internal.so)

target_link_libraries(rec_eos_ops PUBLIC ASC
        ${TF_INTERNAL_LIB}
        )


install(TARGETS rec_eos_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})