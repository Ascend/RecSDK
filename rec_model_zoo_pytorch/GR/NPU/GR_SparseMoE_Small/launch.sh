# 拷贝代码
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout 0626593
#打补丁
cp ../../GR_Base/NPU_GR.patch ./
dos2unix  NPU_GR.patch  #   转为linux格式，打patch报错执行这个
sed -i 's/[ \t]*$//' NPU_GR.patch
git apply NPU_GR.patch
echo "基础补丁加载成功"
cp ../../GR_SparseMOE/NPU_GR_SparseMOE.patch ./
cp ../../GR_SparseMOE/rab_module.patch ./
dos2unix  NPU_GR_SparseMOE.patch  #   转为linux格式，打patch报错执行这个
dos2unix  rab_module.patch  
git apply NPU_GR_SparseMOE.patch
git apply rab_module.patch

cp ../GR_NPU.patch ./
dos2unix  GR_NPU.patch  
git apply GR_NPU.patch

echo "补丁加载成功，可进入examples/hstu执行run_random_2k_xx.sh脚本"

cd examples/hstu
mkdir input_output
