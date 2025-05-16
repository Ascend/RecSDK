cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(feature_map)

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

set(CXX_FLAGS
-O3
-Wall
-fPIC
-fms-extensions
-Wunused-variable
-Wunused-value
-Wcast-align
-Wcast-qual
-Winvalid-pch
-Wwrite-strings
-Wsign-compare
-Wfloat-equal
-Wextra
-D_FORTIFY_SOURCE=2
-std=c++17
-fstack-protector-all
-fstack-protector-strong
)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -ffunction-sections -O3 -Wfatal-errors -DNDEBUG -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "-ldl -rdynamic")

file(GLOB HYBRID_SOURCES "ids_process/*.cpp")
include_directories(ids_process)
include_directories(include)

add_library(hybrid_cpp SHARED bind_torch_lib.cpp ${HYBRID_SOURCES})

target_link_libraries(hybrid_cpp OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}" pthread) # PYTHON SO