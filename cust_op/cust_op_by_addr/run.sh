
#source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /etc/profile
msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
# 截取上一层目录
parent_dir=$(dirname "$msopgen_path")

export PATH=$parent_dir:$PATH

rm -rf ./custom_op
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b -lan cpp -out ./custom_op -m 0 -op EmbeddingLookupByAddress
msopgen gen -i emb_custom.json -f tf -c ai_core-ascend910b -lan cpp -out ./custom_op -m 1 -op EmbeddingUpdateByAddress

cp -rf op_kernel custom_op/
cp -rf op_host custom_op/

cd custom_op

# 判断当前目录下是否存在cmake.json文件
if [ ! -f "CMakePresets.json" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi

#jq '.configurePresets.cacheVariables.ASCEND_CANN_PACKAGE_PATH.value = "/usr/local/Ascend/ascend-toolkit/latest"' CMakePresets.json > tmp.json
#mv tmp.json cmake.json

sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest":g' CMakePresets.json

cd cmake

# 判断当前目录下是否存在config.cmake文件
if [ ! -f "config.cmake" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi


sed -i 's:set(ASCEND_COMPUTE_UNIT ascend910b):set(ASCEND_COMPUTE_UNIT ascend910b ascend910):g' config.cmake

cd ..

bash build.sh

