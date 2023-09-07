# 推理环境部署
1. 下载Tf-serving源码：<strong>https://github.com/tensorflow/serving/archive/1.15.0.zip </strong>
2. 解压后进入源码目录
3. 添加TF-serving第三方依赖
<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp a.执行如下命令，在“serving-1.15.0/third_party”目录下创建“tf_adapter”文件夹并进入。</p>

>  cd third_party/<br>
>  mkdir tf_adapter<br>
>  cd tf_adapter

<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp b. 执行如下命令，在“tf_adapter”文件夹下拷贝存放“libpython3.7m.so.1.0”文件，并创建软链接。</p>

>  cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 .<br>
>  ln -s libpython3.7m.so.1.0 libpython3.7m.so<br>
<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp c. 执行如下命令，在“tf_adapter”文件夹下拷贝存放“_tf_adapter.so”文件，并将“_tf_adapter.so”文件名修改为“lib_tf_adapter.so”。</p>
   
>   cp /usr/local/Ascend/tfplugin/latest/python/site-packages/npu_bridge/_tf_adapter.so .<br>
>   mv _tf_adapter.so lib_tf_adapter.so<br>

4. 编译空的libtensorflow_framework.so、_pywrap_tensorflow_internal.so文件。
<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp a. 在“tf_adapter”文件夹下,vim CMakeLists.txt,写入如下内容保存。

> file(TOUCH \${CMAKE_CURRENT_BINARY_DIR}/stub.c)<br>
> add_library(_pywrap_tensorflow_internal SHARED \${CMAKE_CURRENT_BINARY_DIR}/stub.c)<br>
> add_library(tensorflow_framework SHARED \${CMAKE_CURRENT_BINARY_DIR}/stub.c)<br>

<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp b. 执行:wq!命令保存文件并退出。</p>
<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp c. 执行如下命令，编译出空的.so文件。</p>

> mkdir temp<br>
> cd temp<br>
> cmake ..<br>
> make<br>
> mv lib_pywrap_tensorflow_internal.so ../_pywrap_tensorflow_internal.so<br>
> mv libtensorflow_framework.so ../libtensorflow_framework.so<br>
> cd ..<br>
> ln -s libtensorflow_framework.so libtensorflow_framework.so.1<br>
<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp d. 配置环境命令。</p>

> export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)
5. 在“tf_adapter”文件夹下创建BUILD文件。写入如下内容:

> licenses(["notice"])  # BSD/MIT.<br>
>
>cc_import(<br>
>    name = "tf_adapter",<br>
>    shared_library = "lib_tf_adapter.so",<br>
>    visibility = ["//visibility:public"] <br>
>)<br>
>
>cc_import(<br>
>    name = "tf_python",<br>
>    shared_library = "libpython3.7m.so",<br>
>    visibility = ["//visibility:public"]<br>
> )<br>

6. 修改“serving-1.15.0/tensorflow_serving/model_servers/”路径下的BUILD文件，在“cc_binary”中添加如下加粗内容。
> cc_binary(<br>
>     name = "tensorflow_model_server",<br>
>    stamp = 1,<br>
>    visibility = [<br>
>        ":testing",<br>
>        "//tensorflow_serving:internal",<br>
>    ],<br>
>    deps = [<br>
>        ":tensorflow_model_server_main_lib",<br>
>        <strong>"//third_party/tf_adapter:tf_adapter",<br>
>        "//third_party/tf_adapter:tf_python",<br>
>        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",<br></strong>
>    ],<br>
>)

7. 编译TF Serving。在TF Serving安装目录“serving-1.15.0”下执行如下命令，编译TF Serving。

>bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" tensorflow_serving/model_servers:tensorflow_model_server

8. 建立软连接。
> ln -s /opt/tf_serving/{tf_serving_ID}/execroot/tf_serving/bazel-out/xxx-opt/bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/tensorflow_model_server
- {tf_serving_ID}为一串如“063944eceea3e72745362a0b6eb12a3c”的无规则字符。请根据实际进行填写。
- xxx-opt为工具自动生成文件夹，具体显示请以实际为准。

# 脚本工具介绍
## server.sh / client.sh
启动服务器脚本 / 客户端请求服务器脚本<br>

1. 启动tf-serving server方法<br>
进入目录: tf_serving_inference<br>
更改server.sh中模型路径model_base_path为导出的savedModel路径
执行命令<br>

>将编译tf_serving的第三方依赖tf_adapter路径加入环境变量LD_LIBRARY_PATH<br>
>export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver::/usr/local/python3.7.5/lib:/home/l00832016/serving-1.15.0/third_party/tf_adapter<br>
> sh server.sh<br>
若日志显示Running gRPC ModelServer at 0.0.0.0:xxxx 则表示启动成功
2. 请求服务器方法<br>
执行脚本：sh client.sh<br>
推理成功会打印端到端的时延。<br>

## server_om.sh  / client_om.sh
启动OM模型server / 客户端请求OM模型server
1. 启动OM模型server<br>
进入目录：tf_serving_inference
执行脚本
>export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver::/usr/local/python3.7.5/lib:/home/l00832016/serving-1.15.0/third_party/tf_adapter<br>
>sh server_om.sh, 执行前更改脚本中的模型路径为转化的OM模型路径
2.  请求OM模型<br>
执行脚本 sh client_om.sh