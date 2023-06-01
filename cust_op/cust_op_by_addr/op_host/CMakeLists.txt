
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

add_library(cust_op_proto SHARED ${ops_srcs} ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_link_libraries(cust_op_proto PRIVATE intf_pub exe_graph register)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)

add_library(cust_optiling SHARED ${ops_srcs})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_link_libraries(cust_optiling PRIVATE intf_pub exe_graph register)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
install(TARGETS cust_optiling
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)

