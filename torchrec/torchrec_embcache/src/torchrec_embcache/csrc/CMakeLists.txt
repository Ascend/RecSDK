# Python3
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

# pybind11
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

find_package(Threads REQUIRED)

add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

file(GLOB_RECURSE SOURCES
        ./common/*.cpp
        ./emb_table/*.cpp
        ./embedding_cache/*.cpp
        ./hash_table/*.cpp
        ./feature_filter/*.cpp
        ./ops/*.cpp
        ./utils/*.cpp
        ./file_system/*.cpp
        ./bind.cpp
)

pybind11_add_module(embcache_pybind ${SOURCES})

# 链接 PyTorch 库
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message("TORCH_LIBRARIES:${TORCH_LIBRARIES})")
message("TORCH_PYTHON_LIBRARY:${TORCH_PYTHON_LIBRARY}")

target_link_libraries(embcache_pybind PRIVATE
        pybind11::module
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        securec
)

find_package(OpenMP REQUIRED)
target_link_libraries(embcache_pybind PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(embcache_pybind PRIVATE
        .
)

message("CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")
set_target_properties(embcache_pybind PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS embcache_pybind
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)

if (${CMAKE_BUILD_TYPE} MATCHES "UT")
    message(STATUS "CMAKE_BUILD_TYPE:UT")
    add_subdirectory(ut)
endif ()
