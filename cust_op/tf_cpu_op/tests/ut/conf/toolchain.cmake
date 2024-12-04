option(USE32BIT "Use 32-Bit" OFF)
if (USE32BIT)
    add_compile_options(-m32)
    add_link_options(-m32)
endif()

add_compile_options(-Wall)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)