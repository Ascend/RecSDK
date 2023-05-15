cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE MXREC_OP_SRC ./*.cpp)
add_library(asc_ops SHARED ${MXREC_OP_SRC})
target_link_libraries(asc_ops PUBLIC ASC)
install(TARGETS asc_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})