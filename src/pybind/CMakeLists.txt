cmake_minimum_required(VERSION 3.12)

pybind11_add_module(mxrec_pybind module_main.cpp)
set_target_properties(mxrec_pybind PROPERTIES LINK_FLAGS "-Wl,-rpath,/")
target_link_libraries(mxrec_pybind PUBLIC ASC)
install(TARGETS mxrec_pybind LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})