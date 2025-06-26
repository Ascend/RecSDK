set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/cmake_build/output/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}/embedding_cache/csrc
)

link_libraries(
    securec
    glog
    gtest
    gtest_main
)

add_executable(get_swap_info_test
        get_swap_info_test.cpp
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/embedding_cache/swap_manager.cpp"
        )

# emb_table related
add_executable(emb_mem_pool_test
        emb_table/emb_mem_pool_test.cpp
        ../emb_table/emb_mem_pool.cpp
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/common/common.cpp"
        )

add_executable(emb_table_test
        emb_table/emb_table_test.cpp
        ../emb_table/emb_mem_pool.cpp
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/hash_table/fast_hashmap.cpp"
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/hash_table/hash_bucket.cpp"
        )

# hash_table related
add_executable(hash_bucket_test
        hash_table/net_hash_bucket_test.cpp
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/hash_table/hash_bucket.cpp"
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/common/common.cpp"
        )

add_executable(fast_hashmap_test
        hash_table/fast_hashmap_test.cpp
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/hash_table/fast_hashmap.cpp"
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/hash_table/hash_bucket.cpp"
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/common/common.cpp"
        )

add_executable(feature_filter_test
        feature_filter/feature_filter_test.cpp
        "../../../../../../../../../工作文档/月度文档记录/202506文档/0626 embcache代码同步/修改结果2/csrc/feature_filter/feature_filter.cpp"
        )

# utils related
add_executable(safe_queue_test
        utils/safe_queue_test.cpp
)
target_link_libraries(safe_queue_test PRIVATE Threads::Threads)