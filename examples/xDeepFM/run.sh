kill -9 `ps -ef | grep python | grep -v grep | awk '{print $2}'` > /dev/null 2>&1

# 获取输入参数：py、ip
if [ $# -ge 1 ]; then
  py=$1
  ip=$2
else
  echo "for example: bash run.sh main.py 10.10.10.10 or bash run.sh main.py"
  exit 1
fi

# 检查输入的python文件是否合法
if [[ $py =~ ^[a-z0-9_]+\.py$ ]]; then
  echo "File $py is a valid Python file"
else
  echo "File $py is not a Python file"
  exit 1
fi

# 判断IP地址是否有效
if [ -n "$ip" ]; then
  if [[ $ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    # 将IP地址拆分成四个数字
    ip_array=(${ip//./ })
    # 判断每个数字是否在0-255之间
    valid=true
    for i in "${ip_array[@]}"; do
      if ((i < 0 || i > 255)); then
        valid=false
        break
      fi
    done
    if $valid; then
      echo "ip: $ip is valid"
    else
      echo "ip: $ip is not valid"
      exit 1
    fi
  else
    echo "ip: $ip is not valid."
    exit 1
  fi
fi

cur_path=`pwd`
mx_rec_package_path="/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec" # please config
so_path=${mx_rec_package_path}/libasc
# GLOG_stderrthreshold -2:TRACE -1:DEBUG 0:INFO 1:WARN 2.ERROR, 默认为INFO
mpi_args='-x BIND_INFO="0:12 12:48 60:48" -x GLOG_stderrthreshold=2 -x GLOG_logtostderr=true -bind-to none -x NCCL_SOCKET_IFNAME=docker0 -mca btl_tcp_if_exclude docker0'
interface="lo"
local_rank_size=1 # 每个节点使用的NPU卡数
num_server=1 # 训练节点数
num_process=$((${num_server} * ${local_rank_size})) # 训练总的进程数，等于使用的NPU卡的总数

export HCCL_CONNECT_TIMEOUT=1200 # HCCL集合通信 建链超时时间，取值范围[120,7200]
export PYTHONPATH=${so_path}:$PYTHONPATH # 环境python安装路径
export LD_PRELOAD=/usr/lib64/libgomp.so.1:/usr/local/python3.7.5/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
export LD_LIBRARY_PATH=${so_path}:/usr/local/lib:$LD_LIBRARY_PATH
# 集合通信文件，格式请参考昇腾官网CANN文档，“准备资源配置文件”章节。
export JOB_ID=10086
# 训练任务使用的NPU卡数总数
export MXREC_LOG_LEVEL="ERROR" # 框架日志等级
export TF_CPP_MIN_LOG_LEVEL=3 # tensorflow日志级别,3对应FATAL
# 设置应用类日志的全局日志级别及各模块日志级别，具体请参考昇腾官网CANN文档
export ASCEND_GLOBAL_LOG_LEVEL=3 # “设置日志级别”章节0:debug, 1:info, 2:warning, 3:error, 4:NULL
export MXREC_MODE="ASC"
export USE_MPI=1

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo "Usage: ./run.sh [OPTION]... [IP]..."
    echo " "
    echo "parameter explain:
    [OPTION]       main.py
    [IP]           IP address of the host
    -h/--help		   show help message
    "
    exit 1
fi

# 使用ranktable方案
function rankTableSolution() {
  echo "The ranktable solution"
  export RANK_TABLE_FILE="${cur_path}/hccl_json_${local_rank_size}p.json"
  export RANK_SIZE=$num_process
  export ASCEND_VISIBLE_DEVICES="0"
  export RANK_ID=0
  export ASCEND_DEVICE_ID=$RANK_ID
  echo "RANK_TABLE_FILE=$RANK_TABLE_FILE"
  if [ ! -f "$RANK_TABLE_FILE" ];then
    echo "the rank table file does not exit. Please reference {hccl_json_${local_rank_size}p.json} to correctly config rank table file"
    exit 1
  fi
}

if [ ! -n "$ip" ]; then
  rankTableSolution
else
  VALID_CHECK=$(echo $ip|awk -F. '$1<=255&&$2<=255&&$3<=255&&$4<=255{print "yes"}')
  if echo $ip|grep -E "^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$">/dev/null; then
    if [ "$VALID_CHECK" == "yes" ]; then
      #################使用去除ranktable方案时开启######################
      echo "ip: $ip available."
      echo "The ranktable solution is removed."
      export CM_CHIEF_IP=$ip  # 主节点ip
      export CM_CHIEF_PORT=6000  # 主节点监听端口
      export CM_CHIEF_DEVICE=0  # 主节点device id
      export CM_WORKER_IP=$ip  # 当前节点ip
      export CM_WORKER_SIZE=$num_process  # 参与集群训练的device数量
      echo "CM_CHIEF_IP=$CM_CHIEF_IP"
      echo "CM_CHIEF_PORT=$CM_CHIEF_PORT"
      echo "CM_CHIEF_DEVICE=$CM_CHIEF_DEVICE"
      echo "CM_WORKER_IP=$CM_WORKER_IP"
      echo "CM_WORKER_SIZE=$CM_WORKER_SIZE"
      echo "ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
      #########################################################
    else
      echo "ip: $ip not available!" # 使用ranktable方案
      rankTableSolution
    fi
  else
    echo "ip: $ip not available!" # 使用ranktable方案
    rankTableSolution
  fi
fi

echo "use horovod to start tasks"
DATE=$(date +%Y-%m-%d-%H-%M-%S)
horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" --mpi -H localhost:${local_rank_size} \
python3.7 ${py} 2>&1 | tee "temp_${local_rank_size}p_t_${DATE}.log"
