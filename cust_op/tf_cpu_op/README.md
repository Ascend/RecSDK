# 编译

1、编译release包
+ bash build.sh

2、编译和运行UT：
  (1) bash build.sh ut //编译ut，覆盖率在tests/build/cov/gen目录下
  (2) cd build && bash build_test.sh ut //进入到build目录下并运行ut
+ 编译UT依赖googletest组件，请提准备源码包(trelease-1.8.1版本)，并放置在3rdparty/googletest目录中；

# 在TF中使用
1. 上述运行build.sh，编译通过会在output文件夹下得到rec_base文件，相关头文件以及so包
2. 参考0001-add-sve-op.patch，修改tf源码(打印时间戳的代码请自行删除)；
3. 把lib_rec_base.so拷贝到third_patry/rec_base/lib中；把头文件放到third_patry/rec_base/include中
4. 编译tf成whl包，并安装，命令参考如下：
```shell
bazel build --config=opt ---cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --verbos_failures //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ./out
pip3 install out/tensorfow1.15.0-cp37-cp37m-linux_aarch64.whl --force-reinstall
```
+ 编译和安装过程中需要下载各依赖包，确保环境中可以使用网络；
+ 编译过程中无网络可以提前手动准备好无法下载的包，并使用--distdir参数指定包的路径；
+ 这里使用tf 1.15.0进行验证
+ 各版本依赖：CPU架构支持SVE2,GCC版本10.3.1,python 3.7.5