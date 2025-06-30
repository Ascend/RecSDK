# torchrec-embcache

## 1 拉取依赖
```bash
git submodule update --init --recursive
```

## 2 编译
```bash
bash zbuild_cmake.sh
```
通过cmake进行编译，编译结果在./cmake_build目录下, 其中：
* 编译第三方库(securec/glog/gtest)、embedding_pybind.so都安装在./cmake_build/install目录下;
* UT可执行文件在./cmake_build/output/bin下面;

## 3 编译+打whl包
```shell
bash zbuild.sh
```
本脚本首先会通过cmake进行编译，然后通过setuptools打成whl包，最终可以在./dist目录下查看*.whl包;


## 4 安装和运行
完成Step 3后，可以到dist目录下去安装*.whl包。
```
cd ./dist
pip3 uninstall -y torchrec-embcache
pip3 install torchrec_embcache-0.0.1-py3-none-any.whl
```

安装完成后，可以进行测试.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.0/lib/python3.11/site-packages/torch/lib:/usr/local/python3.11.0/lib/python3.11/site-packages/torchrec_embcache/
cd tests
python3 get_swap_info_test.py
```