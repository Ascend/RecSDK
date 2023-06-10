kill -9 `ps -ef | grep python | grep -v grep | awk '{print $2}'` > /dev/null 2>&1
rm -rf /root/ascend/log/*
rm -rf ./saved-model/*
rm -rf ./kernel*
rm -rf ./export_graph/*

cur_path=`pwd`
mx_rec_package_path="/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec" # please config
so_path=${mx_rec_package_path}/libasc
mpi_args='-x BIND_INFO="0:12 12:48 60:48" -x SPDLOG_LEVEL=debug -bind-to none'
interface="lo"
local_rank_size=8 # 每个节点使用的NPU卡数
num_server=1 # 训练节点数
num_process=$((${num_server} * ${local_rank_size})) # 训练总的进程数，等于使用的NPU卡的总数

export HCCL_CONNECT_TIMEOUT=1200 # HCCL集合通信 建链超时时间，取值范围[120,7200]
export PYTHONPATH=${so_path}:$PYTHONPATH # 环境python安装路径
export LD_PRELOAD=/usr/lib64/libgomp.so.1 # GNU OpenMP动态库路径
export LD_LIBRARY_PATH=${so_path}:/usr/local/lib:$LD_LIBRARY_PATH
# 集合通信文件，格式请参考昇腾官网CANN文档，“准备资源配置文件”章节。
export RANK_TABLE_FILE="${cur_path}/hccl_json_${local_rank_size}p.json"  # 若使用去除ranktable方案，请注释掉这一行
export JOB_ID=10086
# 训练任务使用的NPU卡数总数
export RANK_SIZE=$num_process # 若使用去除ranktable方案，请注释掉这一行
export MXREC_LOG_LEVEL="DEBUG" # 框架日志等级
export TF_CPP_MIN_LOG_LEVEL=3 # tensorflow日志级别,3对应FATAL
# 设置应用类日志的全局日志级别及各模块日志级别，具体请参考昇腾官网CANN文档
export ASCEND_GLOBAL_LOG_LEVEL=3 # “设置日志级别”章节0:debug, 1:info, 2:warning, 3:error, 4:NULL
export MXREC_MODE="ASC"
export USE_MPI=1

################# 参数配置 ######################
export USE_DYNAMIC=0            # 0：静态shape；1：动态shape
export USE_HOT=0                # 0：关闭hot emb；1: 开启hot emb
export USE_DYNAMIC_EXPANSION=0  # 0：关闭动态扩容；1: 开启动态扩容
export USE_MULTI_LOOKUP=1       # 0：一表一查；1：一表多查
export USE_MODIFY_GRAPH=0       # 0：feature spec模式；1：自动改图模式
export USE_TIMESTAMP=0          # 0：关闭特征准入淘汰；1：开启特征准入淘汰
################################################

export KEY_PROCESS_THREAD_NUM=6 #default 6, max 10
#################使用去除ranktable方案时开启######################
#export CM_CHIEF_IP="192.168.1.1"  # 主节点ip
#export CM_CHIEF_PORT=6000  # 主节点监听端口
#export CM_CHIEF_DEVICE=0  # 主节点device id
#export CM_WORKER_IP="192.168.1.1"  # 当前节点ip
#export CM_WORKER_SIZE=$num_process  # 参与集群训练的device数量
#########################################################

py=$1
echo "py is $py"
echo "use horovod to start tasks"
DATE=$(date +%Y-%m-%d-%H-%M-%S)
horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" --mpi -H localhost:${local_rank_size} \
python3.7 ${py} 2>&1 | tee "temp_${local_rank_size}p_${KEY_PROCESS_THREAD_NUM}t_${DATE}.log"

