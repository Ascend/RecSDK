file(GLOB_RECURSE SRCS compute/*.cpp ${REC_BASE_SRC_INCLUDE_DIR}/*.h ${REC_BASE_COMMON_DIR}/*)

add_library(kunpeng OBJECT ${SRCS})
target_link_directories(kunpeng
        PUBLIC
)

target_link_libraries(kunpeng
        -Wl,--start-group
        -Wl,--end-group
        )

target_include_directories(kunpeng
        PUBLIC
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/src/common
)