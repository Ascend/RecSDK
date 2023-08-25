# coding: UTF-8

#  Copyright (C)  2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.7.5/lib
export PATH=$PATH:/usr/local/python3.7.5/bin
export JAVA_HOME=/usr/local/java/
export PATH=${PATH}:${HADOOP_HDFS_HOME}/bin
export PYTHONPATH=`pwd`:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/local/nvidia/cpu_lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${HADOOP_HDFS_HOME}/lib/native:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
export CLASSPATH=`${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob`
export CLASSPATH=.:${CLASSPATH}
