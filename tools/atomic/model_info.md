
### 业务领域/场景
原子操作测试

### 模型框架
TF1.15.0/TF2.6.5

### 使用方法
#### 生成数据集
Python3  gen_mt_data_0to1e.py  5   (这里5的含义为重复度50%)
默认生成在 /home/insert/ 路径下

#### 运行测试
Sparse.sh 需要根据实际环境进行配置
测试 sparse lookup
./sparse.sh  8(卡数)  sparse_lookup.py  8(emb size)  5(重复度)  1 0  0 



