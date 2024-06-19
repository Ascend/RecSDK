# 链接ARM的optimized-routines库
在memcpy等接口占比较大的模型中，有性能收益，源码路径为(https://github.com/ARM-software/optimized-routines/tree/v23.01)
```shell
unzip optimized-routines-23.01.zip
cd optimized-routines-23.01
```

在源码基础上，修改代码，修改脚本如下：
```shell
for m in memcmp memcpy memset memmove memrchr strcpy strchrnul strchr strcmp stpcpy strncmp strnlen strrchr; do
    for f in $(grep __${m}_aarch64 * -r |awk -F ':' '{print $1}'); do
        sed_str1="__${m}_aarch64"
        sed_str2="${m}"
        sed -i 's!'${sed_str1}'!'${sed_str2}'!g' $f
    done
done
```

编译：
```shell
make ARCH=aarch64 -j 8
cp build/lib/libstringlib.so /usr/local/lib/
```

在编译tensorflow serving时链接libstringlib.so，相关修改代码参考0001-Performance-optimization-referrence
运行server时，需要配置环境变量：
```shell
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
```

# 链接jemalloc库
源码下载链接: https://github.com/jemalloc/jemalloc/archive/refs/tags/5.3.0.tar.gz
编译安装命令如下：
```shell
tar -xzvf jemalloc-5.3.0.tar.gz
cd jemalloc-5.3.0
./autogen.sh
make -j 8
make install
```

安装完成后，默认安装在/usr/local/lib/，在编译tensorflow serving时链接libjemalloc.so，相关修改代码参考0001-Performance-optimization-referrence
运行server时，需要配置环境变量：
```shell
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
```

# gRPC配置优化
增加NUM_CQS,MIN_POLLERS,MAX_POLLERS这三个配置项的配置，在多线程请求推理场景可以提升性能
配置项参考gRPC官网(https://grpc.github.io/grpc/cpp/classgrpc_1_1_server_builder.html)
具体修改参考0001-Performance-optimization-referrence，配置最优值根据不同模型和机器可能有所不同；
