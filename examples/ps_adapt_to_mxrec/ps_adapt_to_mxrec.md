# 版本信息

1. ps-lite

   [GitHub - dmlc/ps-lite: A lightweight parameter server interface](https://github.com/dmlc/ps-lite)

   commit 11b42c08a357d4ea5924403daa357587f4d8b5e2（包含本commit及之后都可以）

2. mxRec

   [mxrec: 华为昇腾-MindX 推荐SDK - Gitee.com](https://gitee.com/ascend/mxrec/tree/develop/)

   commit ae36047f1dda8c03fa849184205bdc8bcfb4a137

**注：ps-lite不支持多表存储，所以本文档以单表训练场景为例。**

# 适配流程

## ps-lite

### 下载ps-lite代码

```shell
# 在mxrec根目录下
cd mxrec/src
mkdir 3rdparty
cd 3rdparty
git clone https://github.com/dmlc/ps-lite.git
```

### 修改ps-lite/make/deps.mk

* 调整为不删除源码包，减少重复编译耗时
* 调整依赖版本与ps-lite/CMakeLists.txt一致。其中protobuf 3.8.0为tensorflow 1.15适配版本，用户可根据自身tf版本调整。

```makefile
# protobuf
PROTOBUF = ${DEPS_PATH}/include/google/protobuf/message.h
${PROTOBUF}:
	$(eval FILE=protobuf-cpp-3.8.0.tar.gz)
	$(eval DIR=protobuf-3.8.0)
	rm -rf $(DIR)
	$(WGET) -nc $(URL2)/$(FILE) && tar --no-same-owner -zxf $(FILE)
	cd $(DIR) && export CFLAGS=-fPIC && export CXXFLAGS=-fPIC && ./configure -prefix=$(DEPS_PATH) && $(MAKE) && $(MAKE) install
	rm -rf $(DIR)

# zmq
ZMQ = ${DEPS_PATH}/include/zmq.h

${ZMQ}:
	$(eval FILE=zeromq-4.3.2.tar.gz)
	$(eval DIR=zeromq-4.3.2)
	rm -rf $(DIR)
	$(WGET) -nc $(URL1)/$(FILE) && tar --no-same-owner -zxf $(FILE)
	cd $(DIR) && export CFLAGS=-fPIC && export CXXFLAGS=-fPIC && ./configure -prefix=$(DEPS_PATH) --with-libsodium=no --with-libgssapi_krb5=no && $(MAKE) && $(MAKE) install
	rm -rf $(DIR)

# lz4
LZ4 = ${DEPS_PATH}/include/lz4.h
${LZ4}:
	$(eval FILE=lz4-r129.tar.gz)
	$(eval DIR=lz4-r129)
	rm -rf $(DIR)
	wget -nc $(URL1)/$(FILE) && tar --no-same-owner -zxf $(FILE)
	cd $(DIR) && $(MAKE) && PREFIX=$(DEPS_PATH) $(MAKE) install
	rm -rf $(DIR)

# cityhash
CITYHASH = ${DEPS_PATH}/include/city.h
${CITYHASH}:
	$(eval FILE=cityhash-1.1.1.tar.gz)
	$(eval DIR=cityhash-1.1.1)
	rm -rf $(DIR)
	wget -nc $(URL1)/$(FILE)&& tar --no-same-owner -zxf $(FILE)
	cd $(DIR) && ./configure -prefix=$(DEPS_PATH) --enable-sse4.2 && $(MAKE) CXXFLAGS="-g -O3 -msse4.2" && $(MAKE) install
	rm -rf $(DIR)
```

### 安装依赖

* protobuf：需要确保版本与tensorflow的一致，在tensorflow目录中搜索`GOOGLE_PROTOBUF_VERSION`查看protobuf版本

* zeromq：参考github，版本如ps-lite/make/deps.mk所示

### 准备KVServerMxRecHandle源码

在ps-lite/include/ps/kv_app.h中增加如下代码：

```c++
/**
 * \brief for mxrec embedding storage
 */
template <typename Val>
struct KVServerMxRecHandle {
    void operator()(
      const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
          LL << "KVServerMxRecHandle, customerId:" << req_meta.customer_id << ", push:" << req_meta.push << ", pull:" << req_meta.pull;
          auto es = std::getenv("EMB_SIZE");
          if (es == nullptr) {
              throw std::runtime_error("EMB_SIZE environment variable not found, please export");
          }
          int embeddingSize = std::stoi(es);
          size_t keyCnt = req_data.keys.size();
          KVPairs<Val> res;

          if (req_meta.pull) {
              LL << "pull, customerId:" << req_meta.customer_id << ", keys.size:" << keyCnt << ", embeddingSize:" << embeddingSize;
              res.keys = req_data.keys;
              res.vals.resize(keyCnt * embeddingSize);  // flatten all data
              for (size_t i = 0; i < keyCnt; ++i) {
                  Key key = req_data.keys[i];
                  std::vector<Val> emb = store[key];
                  if (emb.size() == 0) {
                      emb = std::vector<Val>(embeddingSize, 0);
                  } else if (emb.size() != embeddingSize) {
                      throw std::runtime_error("embedding size in server not equal to request");
                  }
                  for (int j = 0; j < embeddingSize; j++) {
                      res.vals[i * embeddingSize + j] = emb[j];
                  }
              }
          } else if (req_meta.push) {
              LL << "push, customerId:" << req_meta.customer_id << ", keys.size:" << keyCnt << ", vals.size:" << req_data.vals.size() << ", embeddingSize:" << embeddingSize;
              for (size_t i = 0; i < keyCnt; i++) {
                  Key key = req_data.keys[i];
                  std::vector<Val> tmp(embeddingSize);
                  for (size_t j = 0; j < embeddingSize; j++)
                  {
                      tmp[j] = res.vals[i * embeddingSize + j];
                  }
                  store[key] = tmp;
              }
          } else {
              LL << "error: request neither push or pull";
              throw std::runtime_error("request neither push or pull");
          }

          server->Response(req_meta, res);
      }
    std::unordered_map<Key, std::vector<Val>> store;
};
```

### 准备scheduler、server、worker源码

* ps-lite/tests/test_scheduler.cc

  ```c++
  #include <cmath>
  #include "ps/ps.h"
  
  using namespace ps;
  
  void RunSchedular(int appId) {
    // start system
    LL << "start schedular, appId:" << appId;
    Start(appId);
    Finalize(appId, true);
    LL << "quit schedular, appId:" << appId;
  }
  
  int main(int argc, char *argv[]) {
    int appId = std::stoi(argv[1]);
    RunSchedular(appId);
    return 0;
  }
  ```

* ps-lite/tests/test_server.cc

  ```c++
  #include <cmath>
  #include "ps/ps.h"
  
  using namespace ps;
  
  void StartServer(int serverId) {
      if (!IsServer()) {
          return;
  	}
      auto server = new KVServer<float>(serverId);
      server->set_request_handle(KVServerMxRecHandle<float>());
      RegisterExitCallback([server](){ delete server; });
  }
  
  void RunServer(int appId) {
      LL << "start server, appId:" << appId;
      Start(appId);
      StartServer(appId);
      // stop system
      Finalize(appId, true);
      LL << "quit server, appId:" << appId;
  }
  
  int main(int argc, char *argv[]) {
      int appId = std::stoi(argv[1]);
      RunServer(appId);
      return 0;
  }
  
  ```

* ps-lite/tests/test_worker.cc

  ```c++
  #include <cmath>
  #include "ps/ps.h"
  
  using namespace ps;
  using std::vector;
  
  
  void RunWorker(int appId, int customerId) {
      LL << "start worker, appId:" << appId << ", customerId:" << customerId; 
      Start(appId);
      if (!IsWorker()) {
          return;
      }
      KVWorker<float> kv(appId, customerId);
  
      // init
      int num = 10000;
      int embSize = 2;
      vector<Key> lens(num, embSize);
      vector<Key> keys(num);
      vector<float> vals(num * embSize);
      int rank = MyRank();
      srand(rank + 7);
      for (int i = 0; i < num; ++i) {
          keys[i] = kMaxKey / num * i + customerId;
          for (int j = 0; j < embSize; ++j)
          {
              vals[i * embSize + j] = rand() % 1000;
          }
      }
  
      // push
      LL << "start push";
      kv.Wait(kv.Push(keys, vals));
  
      // pull
      LL << "start pull";
      std::vector<float> rets;
      kv.Wait(kv.Pull(keys, &rets));
  
      LL << "start validation";
      float res = 0;
      for (int i = 0; i < num; ++i) {
          for (int j = 0; j < embSize; ++j) {
              if (abs(vals[i * embSize + j] - rets[i * embSize + j]) > std::numeric_limits<float>::epsilon()) {
                  LL << "error: embedding from server not equal to original data";
                  Finalize(appId, true);
                  return;
              }
          }
      }
  
      // stop system
      Finalize(appId, true);
      LL << "stop worker, appId:" << appId << ", customerId:" << customerId; 
  }
  
  int main(int argc, char *argv[]) {
      int customerId = std::stoi(argv[1]);
      std::thread t0(RunWorker, 0, customerId);
      t0.join();
      return 0;
  }
  ```

### 修改ps-lite/tests/CMakeLists.txt

修改为如下代码：

```makefile
add_executable(test_schedular test_schedular.cc)
target_link_libraries(test_schedular pslite)

add_executable(test_server test_server.cc)
target_link_libraries(test_server pslite)

add_executable(test_worker test_worker.cc)
target_link_libraries(test_worker pslite)
```

### 修改ps-lite/CMakeLists.txt

增加如下代码：

```cmake
target_link_libraries(pslite PUBLIC pthread)
```

### 编译scheduler、server、worker

在ps-lite目录下执行

```shell
mkdir build
cd build
cmake ..
make -j4
```

### 准备scheduler、server、worker启动脚本

* ps-lite/start_service.sh

    ```shell
    #!/bin/bash
    # set -x
    if [ $# -lt 2 ]; then
        echo "usage: $0 bin_schedular bin_server"
        exit -1;
    fi

    export DMLC_NUM_SERVER=1
    export DMLC_NUM_WORKER=1
    bin_schedular=$1
    bin_server=$2

    # start the scheduler
    export DMLC_PS_ROOT_URI='127.0.0.1'
    export DMLC_ROLE='scheduler'
    export DMLC_PS_ROOT_PORT=8000
    ${bin_schedular} 0 &

    # start servers
    export DMLC_ROLE='server'
    ${bin_server} 0 &

    wait
    ```

* ps-lite/start_worker.sh

  ```shell
  #!/bin/bash
  # set -x
  if [ $# -lt 1 ]; then
      echo "usage: $0 bin_worker"
      exit -1;
  fi
  
  export DMLC_NUM_SERVER=1
  export DMLC_NUM_WORKER=1
  bin_worker=$1
  
  # scheduler info
  export DMLC_PS_ROOT_URI='127.0.0.1'
  export DMLC_PS_ROOT_PORT=8000
  export DMLC_ROLE='worker'
  ${bin_worker} 0 &
  
  wait
  ```


### 编译ps-lite

在ps-lite目录下

```shell
mkdir build
cd build
cmake ..
make -j8
```

### 测试基础功能是否正常

将编译好的test文件复制到ps-lite目录，执行：

```shell
#分别执行
./start_service.sh ./test_schedular ./test_server
./start_worker.sh ./test_worker
```

无报错表示正常。

## mxrec

### 调整ps-lite

1. 删除ps-lite/build
2. 修改ps-lite/CMakeLists.txt，注释掉`add_subdirectory(tests)`

搜索以下代码片段，新增、替换源码。

### src/build.sh

```makefile
cmake -DCMAKE_BUILD_TYPE=Release \
    -DTF_PATH="$1" \
    -DOMPI_PATH="$(whereis openmpi)" \
    -DPYTHON_PATH="$python_path" \
    -DEASY_PROFILER_PATH=/ \
    -DASCEND_PATH="$ascend_path" \
    -DABSEIL_PATH="$1" \
    -DSECUREC_PATH="$2"/../opensource/securec \
    -DCMAKE_INSTALL_PREFIX="$2"/output \
    -DBUILD_CUST="$3" .. \
    -DDEPS_PATH="$2"/src/3rdparty/ps-lite  # new
```

### src/CMakeLists.txt

```cmake
add_subdirecotry(dataset_tf)
add_subdirecotry(core/3rdparty/ps-lite)  # new
```

### src/core/CMakeLists.txt

```cmake
file(GLOB_RECURSE MXREC_SRC ./*.cpp ./*.h)
add_library(ASC SHARED ${MXREC_SRC})

target_include_directories(ASC PUBLIC 3rdparty/ps-lite/include)  # new
```

```makefile
target_link_libraries(ASC PUBLIC ascendcl msprofiler ge_executor gert runtime ge_common register graph ascend_protobuf
    profapi opt_feature error_manager exe_graph acl_tdt_channel acl_tdt_queue securec drvdsmi_host _ock_ctr_common
    pslite  # new
)
```

### src/core/ps_store/ps_store.h**（新增）**

```c
#ifndef MXREC_PS_STORE_H
#define MXREC_PS_STORE_H

#include <map>
#include <memory>

#include "l3_storage/l3_storage.h"
#include "ps/ps.h"  // must set behind any mxrec header file, otherwise will compile fail

using MxRec::L3Storage;
using ps::KVWorker;
using std::map;
using std::shared_ptr;
using std::string;

namespace MxRec {
class PSStore : public L3Storage {
public:
    PSStore(int rankId);

    bool IsTableExist(const string& tableName);

    bool IsKeyExist(const string& tableName, emb_cache_key_t key);

    void CreateTable(const string& tableName, vector<string> savePaths, uint64_t maxTableSize);

    int64_t GetTableAvailableSpace(const string& tableName);

    void InsertEmbeddingsByAddr(const string& tableName, vector<emb_cache_key_t>& keys, vector<float*>& embeddingsAddr,
                                uint64_t extEmbeddingSize);

    void DeleteEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys);

    vector<vector<float>> FetchEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys);

    void Save(int step);

    void Load(const string& tableName, vector<string> savePaths, uint64_t maxTableSize, int step);

    void Start();

    void Stop();

    int64_t GetTableUsage(const string& tableName);

    vector<std::pair<string, vector<emb_cache_key_t>>> ExportTableKey();

private:
    // ps-lite not support multiple table yet, thus this example code only use one client
    int appId = 0;
    int customerId = 0;

    // table --> client
    map<string, std::shared_ptr<KVWorker<float>>> cliMap;
};
}  // namespace MxRec
#endif  // MXREC_PS_STORE_H
```

### src/core/ps_store/ps_store.cpp**（新增）**

```c++
#include "ps_store.h"

using MxRec::PSStore;
using MxRec::emb_cache_key_t;

struct KeyWithIdx {
    emb_cache_key_t key;
    size_t index;
}

bool CompareKeyWithIdx(KeyWithIdx a, KeyWithIdx b) {
    return a.key < b.key;
}

PSStore::PSStore(int rankId) 
{
    this->customerId = rankId + std::stoi(std::getenv("REC_WORKER_ID_START_IDX"));
}

bool PSStore::IsTableExist(const string& tableName)
{
    auto iter = cliMap.find(tableName);
    if (iter == cliMap.end()) {
        return false;
    }
    return true;
}

bool PSStore::IsKeyExist(const string& tableName, emb_cache_key_t key)
{
    auto iter = cliMap.find(tableName);
    if (iter == cliMap.end()) {
        LOG_DEBUG("table:{} not create yet", tableName);
        throw std::runtime_error("table not create yet");
    }

    auto worker = cliMap[tableName];
    vector<emb_cache_key_t> keys = {key};
    vector<float> rets;
    worker->Wait(worker->Pull(keys, &rets));
    if (rets.size() > 0) {
        return true;
    }
    return false;
}

void PSStore::CreateTable(const string& tableName, vector<string> savePaths, uint64_t maxTableSize) {
    static bool alreadyCreate = false;
    if (alreadyCreate) {
        throw runtime_error("ps-lite not support multiple table yet, thus this example code only support one table");
    }
    LOG_DEBUG("start create table:{}, init ps-lite client, appId:{}, customerId:{}", tableName, appId, customerId);
    ps::Start(appId);
    auto worker = make_shared<ps::KVWorker<float>>(appId, customerId);
    cliMap[tableName] = worker;
    LOG_DEBUG("finish create table:{}, worker appId:{}, customerId:{}", tableName, appId, customerId);
    alreadyCreate = true;
}

int64_t PSStore::GetTableAvailableSpace(const string& tableName)
{
    // ps-lite don't have this api
    // thus always available
    return 1000000000000;
}

void PSStore::InsertEmbeddingsByAddr(const string& tableName, vector<emb_cache_key_t>& keys,
                                     vector<float*>& embeddingsAddr, uint64_t extEmbeddingSize)
{
	if (keys.size() == 0) {
        return;
    }

    auto iter = cliMap.find(tableName);
    if (iter == cliMap.end()) {
        LOG_DEBUG("table:{} not create yet", tableName);
        throw std::runtime_error("table not create yet");
    }
    auto psCli = cliMap[tableName];
    
    // note: ps-lite need keys in order
    vector<KeyWithIdx> elements;
    for (size_t i = 0; i < keys.size(); i++) {
        KeyWithIdx e = {keys[i], i};
        elements.push_back(e);
    }
    sort(elements.begin(), elements.end(), CompareKeyWithIdx);
    vector<emb_cache_key_t> sortedKeys;
    vector<float*> sortedEmbeddingsAddr;
    for (size_t i = 0; i < elements.size(); i++) {
		sortedKeys.push_back(elements[i].key);
	    sortedEmbeddingsAddr.push_back(embeddingsAddr[elements[i].index]);
    }
    
    vector<int> lens(keys.size(), extEmbeddingSize);
    vector<float> vals(embeddingsAddr.size() * extEmbeddingSize);
    for (size_t i = 0; i < embeddingsAddr.size(); i++)
    {
        auto rc = memcpy_s(vals.data()+i*extEmbeddingSize, extEmbeddingSize, sortedEmbeddingsAddr[i], extEmbeddingSize);
        if (rc !=0){
            throw std::runtime_error("copy embedding data failed");
        }
    }

    LOG_DEBUG("start push to server, table:{}, keys.size:{}, vals.size:{}", tableName, keys.size(), vals.size());
    int timeStamp = psCli->Push(keys, vals);
    psCli->Wait(timeStamp);

    LOG_DEBUG("end push embedding to server, table:{}", tableName);
}

void PSStore::DeleteEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys) 
{
    LOG_WARN("ps-lite don't have delete function, just return");
    return;
}

vector<vector<float>> PSStore::FetchEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys)
{
    LOG_DEBUG("start pull embedding to server, table:{}, keys.size:{}", tableName, keys.size());
	if (keys.size() == 0) {
        return vector<vector<float>>;
    }
    
    
    auto iter = cliMap.find(tableName);
    if (iter == cliMap.end()) {
        LOG_DEBUG("table:{} not create yet", tableName);
        throw std::runtime_error("table not create yet");
    }
    auto psCli = cliMap[tableName];
    
    // note: ps-lite need keys in order
        vector<KeyWithIdx> elements;
    for (size_t i = 0; i < keys.size(); i++) {
        KeyWithIdx e = {keys[i], i};
        elements.push_back(e);
    }
    sort(elements.begin(), elements.end(), CompareKeyWithIdx);
    vector<emb_cache_key_t> sortedKeys;
    for (size_t i = 0; i < elements.size(); i++) {
		sortedKeys.push_back(elements[i].key);
    }
    
    // input lens will be stuck at req_data.lens, so we use environment variable to work around
    std::vector<float> rets;
    psCli->Wait(psCli->Pull(sortedKeys, &rets));
    
    LOG_DEBUG("finish pull embedding, table:{}, embedding len:{}", tableName, rets.size());
    if (rets.size() % keys.size() != 0) {
        LOG_ERROR("can't split received embedding equally, keys.size:{}, embeddings.size:{}", keys.size(), rets.size());
        throw std::runtime_error("embedding from server incomplete");
    }

    auto extEmbSize = rets.size() % keys.size();
    vector<vector<float>> embs(keys.size());
    for (size_t i = 0; i < elements.size(); i++) {
        auto emb = embs[elements[i].index];
        emb.insert(emb.cbegin(), rets.cbegin() + i * extEmbSize, rets.cend() + (i + 1) * extEmbSize);
    }

    LOG_DEBUG("end pull embedding to server, table:{}", tableName);
    return embs;
}

void PSStore::Save(int step) 
{
    LOG_WARN("ps-lite don't have save function, just return");
}

void PSStore::Load(const string& tableName, vector<string> savePaths, uint64_t maxTableSize, int step) 
{
    LOG_WARN("ps-lite don't have save function, just return");
}

void PSStore::Start() 
{
    LOG_INFO("start ps store");
}

void PSStore::Stop() 
{
    LOG_INFO("start stop ps store");
    ps::Finalize(appId, true);
    LOG_INFO("finish stop ps store");
}

int64_t PSStore::GetTableUsage(const string& tableName)
{
    LOG_WARN("ps-lite don't have GetTableUsage function, just return 0");
    return 0;
}

vector<std::pair<string, vector<emb_cache_key_t>>> PSStore::ExportTableKey()
{
    LOG_WARN("ps-lite don't have export key function, just return empty result");
    return vector<std::pair<string, vector<emb_cache_key_t>>>();
}
```

### src/core/hybrid_mgmt/hybrid_mgmt.cpp

```c++
#include "ps_store/ps_store.h"  // new
```

```c++
if (isL3StorageEnabled) {
    cacheManager = Singleton<MxRec::CacheManager>::GetInstance();
    // 用户可实现L3Storage接口替换SSDEngine以对接外部存储服务
    auto psStore = std::make_shared<PSStore>(mgmtRankInfo.rankId);  // replace
    cacheManager->Init(embCache, mgmtEmbInfo, psStore);  // replace
    EmbeddingMgmt::Instance()->SetCacheManagerForEmbTable(cacheManager);
}
```

### 模型代码

以dcnV2为例，在run.sh中新增以下环境变量。

```shell
# ps-lite info
export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=8  # ausume we run 8 train process

# scheduler info
export DMLC_PS_ROOT_URI='127.0.0.1'  # user can set to remote server
export DMLC_PS_ROOT_PORT=8000

# set role as workers
export DMLC_ROLE='worker'

# mark worker id for train process between multiple train server
# e.g. server A, worker id range [REC_WORKER_ID_START_IDX, +1, ..., +n]; server B, worker id range [REC_WORKER_ID_START_IDX +(n+1), +(n+2), ...]
export REC_WORKER_ID_START_IDX=0
```

在ps-lite目录拉起存储服务

```shell
./start_service.sh ./test_schedular ./test_server
```

在模型目录拉起训练

```shell
# 修改缓存模式为SSD（按上述mxrec源码修改步骤，SSDEngine已被替换为ps-lite，为了不影响对外接口，未修改对外暴露的ssd参数，用户可自行修改）
export CACHE_MODE="SSD"

./run.sh $LIBSAC_PATH $PYTHON_PATH $HCCL_JSON_PATH $DATA_PATH
```





