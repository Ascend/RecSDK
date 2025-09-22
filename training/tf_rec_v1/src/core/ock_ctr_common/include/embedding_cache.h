/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.s
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef EMBEDDING_CACHE_H
#define EMBEDDING_CACHE_H

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace EmbCache {
using KeyOffsetPair = std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;

struct EmbBaseInfo {
    EmbBaseInfo(int batchId, int channelId, std::string name, bool isDp, bool paddingKeysMask,
                std::vector<int64_t> paddingKeys)
        : batchId(batchId),
          channelId(channelId),
          name(name),
          isDp(isDp),
          paddingKeysMask(paddingKeysMask),
          paddingKeys(paddingKeys)
    {
    }

    int batchId;
    int channelId;
    std::string name;
    bool isDp{false};
    bool paddingKeysMask{false};
    std::vector<int64_t> paddingKeys;
};

class Initializer {
public:
    Initializer() = default;
    virtual ~Initializer() = default;

    /* *
     * 生成随机数
     * @Param emb embedding的首地址
     */
    virtual void GenerateData(float* emb, int embSize) = 0;
    uint32_t start = 0;     // 起始位置
    uint32_t len = 0;       // 初始化的长度
    float initParam = 1.0;  // 初始化器生成的初始值均需要乘以initParam
};

enum class InitializerType {
    INVALID,
    CONSTANT,
    TRUNCATED_NORMAL,
    RANDOM_NORMAL
};

struct ConstantInitializerInfo {
    ConstantInitializerInfo() = default;

    ConstantInitializerInfo(float constantValue, float initK);

    float constantValue = 0;  // 常量值
    float initK = 1.0;        // 初始化出来的值需乘以initK
};

struct NormalInitializerInfo {
    NormalInitializerInfo() = default;

    NormalInitializerInfo(float mean, float stddev, uint32_t seed, float initK);

    float mean = 0;     // 平均值
    float stddev = 0;   // 标准差
    uint32_t seed = 0;  // 随机数种子
    float initK = 1.0;  // 初始化出来的值需乘以initK
};

class ConstantInitializer : public Initializer {
public:
    ConstantInitializer() = default;

    ConstantInitializer(uint32_t start, uint32_t len, float value, float initK);

    ~ConstantInitializer() override = default;

    void GenerateData(float* emb, int embSize) override;

    uint32_t start = 0;       // 起始位置
    uint32_t len = 0;         // 初始化的长度
    float constantValue = 0;  // 常量值
};

class RandomNormalInitializer : public Initializer {
public:
    RandomNormalInitializer() = default;
    RandomNormalInitializer(uint32_t start, uint32_t len, NormalInitializerInfo& initInfo);

    ~RandomNormalInitializer() override = default;

    void GenerateData(float* emb, int embSize) override;

    uint32_t start = 0;  // 起始位置
    uint32_t len = 0;    // 初始化的长度
    float mean = 0;      // 平均值
    float stddev = 0;    // 标准差
    uint32_t seed = 0;   // 随机数种子

    std::default_random_engine generator;          // 随机数生成器
    std::normal_distribution<float> distribution;  // 正态分布
};

class TruncatedNormalInitializer : public Initializer {
public:
    TruncatedNormalInitializer() = default;

    TruncatedNormalInitializer(uint32_t start, uint32_t len, NormalInitializerInfo& initInfo);

    ~TruncatedNormalInitializer() override = default;

    void GenerateData(float* emb, int embSize) override;

    int boundNum = 2;

    uint32_t start = 0;  // 起始位置
    uint32_t len = 0;    // 初始化的长度
    float mean = 0;      // 平均值
    float stddev = 0;    // 标准差
    uint32_t seed = 0;   // 随机数种子

    std::default_random_engine generator;  // 随机数生成器
    std::normal_distribution<float> distribution;
    float minBound = 0;  // 下界
    float maxBound = 0;  // 上界
};

struct InitializerInfo {
    InitializerInfo() = default;

    InitializerInfo(std::string& name, uint32_t start, uint32_t len, ConstantInitializerInfo constantInitializerInfo);

    InitializerInfo(std::string& name, uint32_t start, uint32_t len, NormalInitializerInfo normalInitializerInfo);

    std::string name = "";  // 初始化器的名称
    uint32_t start = 0;     // 初始化开始的位置
    uint32_t len = 0;       // 待初始化的长度
    InitializerType initializerType = InitializerType::INVALID;

    ConstantInitializerInfo constantInitializerInfo;
    NormalInitializerInfo normalInitializerInfo;

    std::shared_ptr<Initializer> initializer;
};

struct EmbCacheInfo {
    EmbCacheInfo(std::string tableName, uint64_t vocabSize, uint32_t embeddingSize, uint32_t extEmbeddingSize,
                 uint64_t maxCacheSize)
        : tableName(tableName),
          vocabSize(vocabSize),
          embeddingSize(embeddingSize),
          extEmbeddingSize(extEmbeddingSize),
          maxCacheSize(maxCacheSize)
    {
    }
    std::string tableName = "";
    uint64_t vocabSize = 0;  // host侧的容量(能存多少条embedding)
    uint32_t embeddingSize = 0;
    uint32_t extEmbeddingSize = 0;  // 包含embedding和优化器信息的embedding长度
    uint64_t maxCacheSize = 0;      // device侧的容量(能存多少条embedding)
};

class EmbCacheManager {
public:
    virtual ~EmbCacheManager() = default;

    /* *
     * 对当前embInfo对应的table在cache_manager中进行table初始化
     * @Param EmbCacheInfo: embedding cache的初始化信息
     * @Param std::vector<InitializerInfo> 初始化器的信息
     * @Param uint64_t prefillBufferSize emb内存池恒定可用大小
     * @Param uint32_t refillThreadNum emb内存池自动填充线程数
     * @Return errorCode
     */
    virtual int CreateCacheForTable(const EmbCacheInfo& embCacheInfo,
                                    const std::vector<InitializerInfo>& initializerInfos,
                                    int64_t invalidKey = DEFAULE_INVALID_KEY,
                                    uint64_t prefillBufferSize = DEFAULE_PREFILL_BUFFER_SIZE,
                                    uint32_t refillThreadNum = DEFAULE_REFILL_THREAD_NUM) = 0;

    /* *
     * 查找当前keys对应的offsets并将本不存在与offsetMapper中的keys插入到offsetMapper中并得到其偏移值offsets，
     * 并且当offsetMapper可存放空间不足时，释放可swapOut的keys，获取当前需要被换入换出的keys和offsets的pair
     * @Param info: The EmbBaseInfo instance.
     * @Param keys: 当前batch所有unique的keys
     * @Param swapInKoPair: 输出参数，需要换入的Key-offset pair
     * @Param swapOutKoPair: 输出参数，需要换出的Key-offset pair
     * @Return errorCode
     */
    virtual int GetSwapPairsAndKey2Offset(const EmbBaseInfo& info, std::vector<uint64_t>& keys,
                                          KeyOffsetPair& swapInKoPair, KeyOffsetPair& swapOutKoPair) = 0;

    /* *
     * 查询Embedding的地址
     * @Param tableName: 表名
     * @Param keys: 待查询的keys
     * @Param addrs: keys对应的申请出来存放embedding的空间首地址
     * @Param threadNum: 线程数
     * @Return errorCode
     */
    virtual int EmbeddingLookupAddrs(const std::string& tableName, const std::vector<uint64_t>& keys,
                                     std::vector<float*>& addrs, uint32_t threadNum = DEFAULE_LOOKUP_THREAD_NUM) = 0;

    /* *
     * 更新Embedding
     * @Param tableName: 表名
     * @Param keys: 待更新的keys，用于查询出每个key在DDR上存放的地址
     * @Param embAddr: 待更新到DDR上的embedding的首地址
     * @Param threadNum: 线程数
     * @Return errorCode
     */
    virtual int EmbeddingUpdate(const std::string& tableName, const std::vector<uint64_t>& keys, float* embAddr,
                                uint32_t threadNum = DEFAULE_LOOKUP_THREAD_NUM) = 0;

    /* *
     * 在EmbLocalTable中移除keys，并将存储其embedding的内存位置记为可复用
     * @Param tableName: 表名
     * @Param keys: 待移除的keys
     * @Return errorCode
     */
    virtual int EmbeddingRemove(const std::string& tableName, const std::vector<uint64_t>& keys,
                                uint32_t threadNum = DEFAULE_LOOKUP_THREAD_NUM) = 0;

    /* *
     * 将需要被淘汰的keys从offsetMapper的记录中移除，同时也在EmbLocalTable中移除，并将存储其embedding的内存位置记为可复用
     * @Param tableName: 表名
     * @Param keys: 待淘汰的keys
     * @Return errorCode
     */
    virtual int RemoveEmbsByKeys(const std::string& tableName, const std::vector<uint64_t>& keys) = 0;

    /* *
     * 获取所有table names
     * @Param allTableNames: 输出参数，用于存放所有的table names
     * @Return errorCode
     */
    virtual int GetEmbTableNames(std::vector<std::string>& allTableNames) = 0;

    /* *
     * 获取以values为增序排列的当前记录在offsetMapper中所有的keys和values的pairs
     * @Param tableName: 表名
     * koVec: 输出参数
     * @Return errorCode
     */
    virtual int ExportDeviceKeyOffsetPairs(const std::string& tableName,
                                           std::vector<std::pair<uint64_t, uint64_t>>& koVec) = 0;

    /* *
     * 获取当前table的序列化信息
     * @Param tableName: 要序列化的表
     * @Param buffer: 输出参数，存储序列化之后的信息
     * @Return errorCode
     */
    virtual int Serialize(const std::string& tableName, std::vector<char>& buffer) = 0;

    /* *
     * 将当前table的序列化信息进行反序列化
     * @Param tableName: 要反序列化的表
     * @Param buffer: 输入参数，将buffer中的内容进行反序列化
     * @Return errorCode
     */
    virtual int Deserialize(const std::string& tableName, const std::vector<char>& buffer) = 0;

    /* *
     * 析构所有embCache，释放内存
     */
    virtual void Destroy() = 0;

    /* *
     * 查询表的使用量
     * @Param tableName: 要查询的表
     * @Return 当前表的使用量
     */
    virtual uint32_t GetUsage(const std::string& tableName) = 0;

    /* *
     * 获取当前host侧所存储的所有keys及其对应的embeddings和优化器参数
     * @Param tableName: 需要获取信息的table名字
     * @Param keys: 输入参数，输入空vector，获取的存储的所有keys会赋到该vector中
     * @Param embeddings: 输入参数，输入空vector，获取的存储的所有embeddings会赋到该vector中
     * @Param optimizerSlots: 输入参数，输入空vector，获取的存储的所有optimizerSlots会赋到该vector中
     * @Return errorCode
     */
    virtual int GetEmbTableInfos(std::string tableName, std::vector<uint64_t>& keys,
                                 std::vector<std::vector<float>>& embeddings,
                                 std::vector<std::vector<float>>& optimizerSlots) = 0;

    /* *
     * 将所需存储的keys及其对应的embeddings和优化器参数传入，来装载LocalEmbeddingTable
     * @Param tableName: 需要加载信息的table名字
     * @Param keys: 输入参数，需要加载的所有keys
     * @Param embeddings: 输入参数，需要加载的所有embeddings
     * @Param optimizerSlots: 输入参数，需要加载的所有optimizerSlots
     * @Return errorCode
     */
    virtual int LoadEmbTableInfos(std::string tableName, const std::vector<uint64_t>& keys,
                                  const std::vector<std::vector<float>>& embeddings,
                                  const std::vector<std::vector<float>>& optimizerSlots) = 0;

    /* *
     * When switch the channel to eval, backup the current table's offsetMapper object.
     * @Param tableName: embedding table name
     * @Return errorCode
     */
    virtual int BackUpTrainStatus(const std::string& tableName) = 0;

    /* *
     * When switch the eval channel back to train, Recover the current table's offsetMapper object to the backup state.
     * @Param tableName: embedding table name
     * @Return errorCode
     */
    virtual int RecoverTrainStatus(const std::string& tableName) = 0;

    /* *
     * Reset the offsetMapper object to revert to its initialized state after loading.
     * @Return errorCode
     */
    virtual int ResetOffsetMappers() = 0;

    /* *
     * Gets the padding keys offset of the table.
     * @Param tableName: The embedding table name.
     * @Return The padding keys offset.
     */
    virtual std::unordered_set<uint64_t> GetPaddingKeysOffset(const std::string& tableName) = 0;

private:
    static constexpr uint32_t DEFAULE_LOOKUP_THREAD_NUM = 4;
    static constexpr uint32_t DEFAULE_REFILL_THREAD_NUM = 4;
    static constexpr uint64_t DEFAULE_PREFILL_BUFFER_SIZE = 500000;
    static constexpr int64_t DEFAULE_INVALID_KEY = -1;
};
}  // namespace EmbCache

#endif  // EMBEDDING_CACHE_H
