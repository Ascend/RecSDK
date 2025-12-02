# 拷贝代码
git config --global http.sslVerify false
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout 0626593
#打补丁
cp ../../GR_Base/NPU_GR.patch ./
dos2unix  NPU_GR.patch  #   转为linux格式，打patch报错执行这个
sed -i 's/[ \t]*$//' NPU_GR.patch
git apply NPU_GR.patch
echo "基础补丁加载成功"
cp ../NPU_GR_SparseMOE.patch ./
dos2unix  NPU_GR_SparseMOE.patch  #   转为linux格式，打patch报错执行这个
sed -i 's/[ \t]*$//' NPU_GR.patch
git apply NPU_GR_SparseMOE.patch

echo "补丁加载成功，可进入examples/hstu执行run_random_2k.sh脚本"

cd examples/hstu
mkdir input_output
