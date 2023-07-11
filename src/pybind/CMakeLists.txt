cmake_minimum_required(VERSION 3.20)

pybind11_add_module(mxrec_pybind module_main.cpp)
set_target_properties(mxrec_pybind PROPERTIES LINK_FLAGS "-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")
target_include_directories(mxrec_pybind PUBLIC ${ASCEND_DRIVER_PATH}/include)
target_link_directories(mxrec_pybind PUBLIC ${ASCEND_DRIVER_PATH}/lib64/driver)
target_link_libraries(mxrec_pybind PUBLIC ASC)
install(TARGETS mxrec_pybind LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
