# 介绍
推荐模型中，部分算子如（select等）在NPU上亲和性差，会存在切分到CPU上进行运算的场景，所以针对tf的cpu算子，用sve指令集进行性能优化；
sve相关资料可以查看[Introduction to SVE](https://developer.arm.com/documentation/102476/0100?lang=en)和[ARM C Language Extensions for SVE](https://developer.arm.com/documentation/100987/0000/?lang=en)

使用模板特化方法，使得某些固定类型的入参（入参维度均为1维）会走到本次sve指令优化的算子中，涉及的算子及路径如下：

| 算子名称   | 特化类型                | 代码相对路径                                      |
|-----------|-----------------------|-------------------------------------------------|
| less      | tf.int32,tf.int64     | src/kunpeng/compute/cmp/cmp_sve.cpp             |
| greater   | tf.int32,tf.int64     | src/kunpeng/compute/cmp/cmp_sve.cpp             |
| floormod  | tf.float32,tf.float64 | src/kunpeng/compute/floor_mod/floor_mod_sve.cpp |
| select    | tf.int64              | src/kunpeng/compute/select/select_sve.cpp       |

# 算子使用流程
1.编译sve算子包  
2.修改tf源码，把算子包打包链接到tf中  
3.编译安装修改后的tf  
4.正常使用上表中的四个算子即可   

本文样例运行环境部分组件版本如下：
+ CPU架构须支持SVE2
+ GCC版本为10.3.1或以上
+ python 3.7.5
+ bazel 0.24.1

# 编译sve算子包
## 编译release包 
```shell
bash build.sh
```
编译通过会在当前目录的output/rec_base/lib目录中生成lib_rec_base.so，后续编译tf需要将so拷贝到tf的目录中，调用算子需要将lib的目录配置到环境变量中;

## 编译ut测试包并运行ut
会使用到gtest组件，所以编译ut之前，请提前准备好gtest的源码包（1.8.1版本），放置到3rdparty/googletest目录中；
1. bash build.sh ut // 编译ut
2. cd build && bash build_test.sh ut

# 安装TF
+ 编译和安装过程中需要下载各依赖包，确保环境中可以正常使用网络
+ TF编译过程如个别包无法下载，可以手动下载后，放置到环境上，并通过--distdir参数指定包的路径
+ 这里使用tf1.15.0进行验证
## 编译安装bazel
安装bazel 0.24.1
参考昇腾官网文档[源码安装0.24.1版本bazel](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/800alpha002/migration/tfmigr1/atlastfserv_26_0008.html)章节
其中解压开bazel文件之后，进入到目录中，需要修改代码，命令如下：
```shell
sed -i '21a #include <limits>' third_party/ijar/zlib_client.cc
sed -i '23a #include <limits>' third_party/ijar/mapped_file_unix.cc
sed -i '43a }' third_party/grpc/src/core/lib/gpr/log_linux.cc
sed -i '42a namespace {' third_party/grpc/src/core/lib/gpr/log_linux.cc
sed -i 's/gettid()/::gettid()/g' third_party/grpc/src/core/lib/gpr/log_linux.cc
```

## 编译安装TF
1. 参考昇腾官网[安装开源框架tensoeflow](https://www.hiascend.com/document/detail/zh/TensorFlowCommercial/80RC3/migration/tfmigr1/tfmigr1_000008.html)按照aarch64架构说明，执行其中的安装h5py；下载tensorflow源码包；下载nsync源码包，修改nsync后重新打包并修改sha256sum步骤操作；
2. 参考0001-add-sve-op.patch修改tf源码
3. 将lib_rec_base.so拷贝到tf的third_party/rec_base/lib目录中（其中lib目录需要自己创建）
4. 运行如下命令生成**.tf_configure.bazelrc**配置文件
```shell
chmod +x configure
./configure
```
5. 运行如下命令编译TF
```shell
chmod 777 ./tensorflow/tools/pip_package/build_pip_package.sh
bazel build --config=opt --distdir=/xxx/xxx --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --verbose_failures //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./out
```
其中--distdir参数可选，如所有包都正常下载，则无需配置；  
如果编译中出现如下报错需要修改用户目录下.cache中的三个文件，其中xxx为用户名，path_xxx为编译自动生成路径
“/xxx/.cache/bazel/_bazel_root/path_xxx/external/hwloc/BUILD.bazel:229:1: C++ compilation of rule '@hwloc//:hwloc' failed (Exit 1): gcc failed: error executing command”
修改命令如下：
```shell
sed -i '45d' /xxx/.cache/bazel/_bazel_root/path_xxx/external/hwloc/hwloc/topology.c
sed -i '43a }' /xxx/.cache/bazel/_bazel_root/path_xxx/external/grpc/src/core/lib/gpr/log_linux.cc
sed -i '42a namespace {' /xxx/.cache/bazel/_bazel_root/path_xxx/external/grpc/src/core/lib/gpr/log_linux.cc
sed -i 's/gettid()/::gettid()/g' /xxx/.cache/bazel/_bazel_root/path_xxx/external/grpc/src/core/lib/gpr/log_linux.cc
sed -i '42a #include <limits>' /xxx/.cache/bazel/_bazel_root/path_xxx/external/com_google_absl/absl/synchronization/internal/graphcycles.cc
```
6. 安装tf
```shell
pip3 install out/tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl
```
tips:这一步会下载安装依赖包，请确保网络正常

# 运行demo
配置lib_rec_base.so所在目录到LD_LIBRARY_PATH环境变量中：export LD_LIBRARY_PATH=/xx/xx:$LD_LIBRARY_PATH  
进入到demo路径下  
python3 demo.py --op=Less --iters=5000 --minval=5000 --maxval=10000 --row=1 --col=4000

入参解释：  
--op:指定算子类型，范围为Less，Greater，FloorMod，Select  
--iters：指定demo测试算子的次数，整型,范围为[1-10000]  
--minval:用于算子计算的随机数的最小值，整型,范围为[-32767，32768]，并且小于maxval  
--maxval:用于算子计算的随机数的最大值，整型,范围为[-32767，32768]，并且大于minval  
--row，--col：用于算子计算的数据的shape，row只支持1，col为整型,取值范围[1,100000]  