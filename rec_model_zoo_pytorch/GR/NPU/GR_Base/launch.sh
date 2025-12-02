# 拷贝代码
git config --global http.sslVerify false
git clone https://github.com/NVIDIA/recsys-examples.git
cd recsys-examples && git checkout 0626593
# 打补丁
cp ../NPU_GR.patch ./
dos2unix NPU_GR.patch
sed -i 's/[ \t]*$//' NPU_GR.patch
git apply NPU_GR.patch

echo "补丁加载成功，可进入recsys-examples/examples/hstu目录,执行run_random_2k.sh脚本"

cd examples/hstu
mkdir input_output
