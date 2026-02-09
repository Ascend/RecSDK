# FAQ<a name="ZH-CN_TOPIC_0000001579847284"></a>

## 多机训练HCCL集群通信失败<a name="ZH-CN_TOPIC_0000001795546622"></a>

**问题现象<a name="section964994911912"></a>**

HCCL集群通信失败。

**可能原因<a name="section1581817491790"></a>**

-   多机节点的NPU device ip不能互相ping通。
-   多机节点的NPU device的TLS配置不同。
-   其他

**解决方案<a name="section181445017916"></a>**

1.  检查多机节点的NPU device ip是否能互相ping通。以双机集群（节点A、B），每个节点8卡示例。
    1.  查询节点A的device ip:

        ```bash
        for i in {0..7}; do hccn_tool -i $i -ip -g ; done
        ```

    2.  在节点B上ping节点A的device ip：

        ```bash
        hccn_tool -i 0 -ping -g address 192.x.x.x
        ```

        其中192.x.x.x为节点A的rank0的device ip；0为指定使用B节点rank0 device去ping对应ip。

        若指令回显包含“0.00% packet loss”则说明能ping通；ping不通则需检查环境网络配置；

        >[!NOTE] 说明 
        >若device ip配置为IPv6，查询device ip指令和ping device指令有所区别，示例：
        >-   查询device ip：
        >    ```bash
        >    for i in {0..7}; do hccn_tool -i $i -ip -inet6 -g; done
        >    ```
        >-   ping指定device ip：
        >    ```bash
        >    hccn_tool -i 0 -ping -inet6 -g ipv6_address x:x:x:x
        >    ```

2.  检查多机节点的NPU device的TLS配置是否相同。

    在两个节点上分别执行如下指令，查看配置是否相同：

    ```bash
    for i in {0..7}; do hccn_tool -i $i -tls -g  |grep switch; done
    ```

    若配置不同，则需修改TLS配置。

    ![](../../figures/tf_rec_v1/zh-cn_image_0000001795706422.png)

    TLS状态开关设置和证书信息修改的详细的方法请参照《HCCL集合通信库用户指南》的“参与集合通信的服务器TLS信息不一致，HCCL初始化失败”章节。

3.  其他集合通信相关问题可参考《HCCL集合通信库用户指南》的“常见问题处理”章节。


## 在ARM环境下，使用Rec SDK TensorFlow进行模型训练报错<a name="ZH-CN_TOPIC_0000001809551534"></a>

**问题现象<a name="section964994911912"></a>**

在ARM环境下，使用Rec SDK TensorFlow进行模型训练并且导入了scikit-learn库时存在报错：`ImportError: /usr/local/python3.7.5/lib/python3.7/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block`。

**可能原因<a name="section1581817491790"></a>**

Rec SDK TensorFlow编译使用了OpenMP，OpenMP将使用Thread Local Storage（动态TLS）内存空间，sklearn在执行一些并行计算的时候需要使用静态TLS空间。在aarch64架构的机器上动态TLS和静态TLS使用的是相同的预分配池，先导入Rec SDK TensorFlow时导致预分配过多的内存空间，导致sklearn导入时libgomp.so空间不足。

**解决方案<a name="section181445017916"></a>**

在模型代码中的main.py文件中将import sklearn置于导入Rec SDK TensorFlow之前，保证libgomp.so有足够的静态TLS空间。

## 使用glibc 2.17的镜像训练模型时出现Core Dump问题<a name="ZH-CN_TOPIC_0000002444888557"></a>

**问题现象<a name="section76445285718"></a>**

在使用基础镜像为Centos7.6的环境跑推荐模型时，遇到如下堆栈问题。

![](../../figures/tf_rec_v1/zh-cn_image_0000002411341900.png)

**可能原因<a name="section7693101619818"></a>**

glibc 2.17在处理TLS（Thread-Local Storage, 线程局部存储）时，大量dlopen、dlclose、pthread_create并发执行可能会导致`_dl_allocate_tls_init`段错误。详细信息请参考[链接](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)中的问题根因、测试代码、修复代码。

**解决方案<a name="section1033624316810"></a>**

1.  glibc 2.34版本已修复该问题。建议用户使用glibc 2.34及以上版本训练模型。
2.  参考[链接](https://sourceware.org/bugzilla/show_bug.cgi?id=19329)中的修复代码，修复训练环境中的glibc。
3.  尝试执行命令：**export LD\_PRELOAD=/usr/lib64/libstdc++.so.6**。


