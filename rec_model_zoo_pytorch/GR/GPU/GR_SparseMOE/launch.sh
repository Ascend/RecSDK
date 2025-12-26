# 拷贝代码
git config --global http.sslVerify false
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout 0626593
# 打补丁
cp ../../GR_Base/GPU_GR.patch ./
dos2unix  GPU_GR.patch  #   转为linux格式，打patch报错执行这个
sed -i 's/[ \t]*$//' GPU_GR.patch
git apply GPU_GR.patch
echo "基础补丁加载成功"
cp ../GPU_GR_SparseMOE.patch ./
cp ../rab_module.patch ./
dos2unix  GPU_GR_SparseMOE.patch  #   转为linux格式，打patch报错执行这个
dos2unix  rab_module.patch  
git apply GPU_GR_SparseMOE.patch
git apply rab_module.patch

echo "补丁加载成功，可进入recsys-examples/hstu目录,执行run_random_2k.sh脚本"

cd examples/hstu
mkdir input_output
