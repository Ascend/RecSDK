# 安全加固<a name="ZH-CN_TOPIC_0000001580166480"></a>

## 安全要求<a name="ZH-CN_TOPIC_0000001580007128"></a>

### 文件权限<a name="ZH-CN_TOPIC_0000001862966101"></a>

使用API读取文件时，用户需要保证该文件的owner必须为自己，且权限不大于640，避免发生提权等安全问题。

外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。


### Hadoop安装权限控制<a name="ZH-CN_TOPIC_0000001816046302"></a>

如需使用Hadoop分布式文件系统，在完成了Hadoop的安装之后，请控制安装目录下的libhdfs.so文件权限不大于640，防止so包被篡改产生命令注入的风险。



## 加固须知<a name="ZH-CN_TOPIC_0000001580007116"></a>

本文中列出的安全加固措施为基本的加固建议项。用户应根据自身业务，重新审视整个系统的网络安全加固措施，必要时可参考业界优秀加固方案和安全专家的建议。


## 操作系统安全加固<a name="ZH-CN_TOPIC_0000001580326416"></a>

### 防火墙配置<a name="ZH-CN_TOPIC_0000001629887053"></a>

操作系统安装后，若配置普通用户，可以通过在“/etc/login.defs”文件中新增“ALWAYS\_SET\_PATH=yes”配置，防止越权操作。


### 设置umask<a name="ZH-CN_TOPIC_0000001630127045"></a>

建议用户将宿主机和容器中的umask设置为027及其以上，提高文件权限。

以设置umask为027为例，具体操作如下所示。

1.  以root用户登录服务器，编辑“/etc/profile”文件。

    ```bash
    vim /etc/profile
    ```

2.  在“/etc/profile”文件末尾加上**umask 027**，保存并退出。
3.  执行如下命令使配置生效。

    ```bash
    source /etc/profile
    ```


### 无属主文件安全加固<a name="ZH-CN_TOPIC_0000001630246517"></a>

因为官方Docker镜像与物理机上的操作系统存在差异，系统中的用户可能不能一一对应，导致物理机或容器运行过程中产生的文件变成无属主文件。

用户可以执行**find / -nouser -o -nogroup**命令，查找容器内或物理机上的无属主文件。根据文件的UID和GID创建相应的用户和用户组，或者修改已有用户的UID、用户组的GID来适配，赋予文件属主，避免无属主文件给系统带来安全隐患。


### 端口扫描<a name="ZH-CN_TOPIC_0000001630127077"></a>

需要关注全网侦听的端口和非必要端口，如有非必要端口请及时关闭。建议用户关闭不安全的服务，如Telnet、FTP等。具体关闭方法请参考所使用的操作系统相关文档。


### 防DoS攻击<a name="ZH-CN_TOPIC_0000001630046413"></a>

用户可以通过实现IP限制和服务器速率限制对系统进行防DoS攻击，方法包括但不限于利用Linux系统自带iptables防火墙进行预防、优化sysctl参数等。具体使用方法，用户可自行查阅相关资料。


### 防止MPI全零侦听<a name="ZH-CN_TOPIC_0000001676600585"></a>

-   MPI初始化拉起的后台mpirun进程存在端口未认证问题，存在被攻击的风险。

    规避措施：MPI启动之后添加防火墙规则，防止外部网络连接mpirun和orted（mpi内部组件）侦听的端口，运行结束后需要清理防火墙规则。

    -   添加防火墙规则

        ```bash
        ```shell
        iptables -D INPUT -p tcp -j ${规则名}
        iptables -F ${规则名}
        iptables -X ${规则名}
        iptables -t filter -N ${规则名}
        iptables -I INPUT -p tcp -j ${规则名}
        iptables -t filter -I ${规则名} -i ${要限制网卡} -p tcp --dport ${要限制的端口} -j DROP
        ```
  


    -   清除防火墙规则

        ```bash
        ```shell
        iptables -D INPUT -p tcp -j ${规则名}
        iptables -F ${规则名}
        iptables -X ${规则名}
        ```


-   MPI默认配置项在启动多个Python训练进程时会对0.0.0.0地址进行侦听。

    规避措施：找到MPI配置路径下的etc文件夹，在“openmpi-mca-params.conf”文件末尾添加以下两行配置项。

    ```bash
    btl_tcp_if_include = lo
    btl_tcp_if_exclude = docker0
    ```

    >[!NOTE] 说明 
    >推荐用户仅在容器中使用MPI，并由网络命名空间进行隔离。以保证宿主机上无全零侦听，且宿主机其他用户无法连接OpenMPI未认证端口。




## Device安全加固<a name="ZH-CN_TOPIC_0000002072744845"></a>

Device侧OS在申请大页内存后，默认会对大页内存做清零操作，与此同时，性能会下降。用户如果需要高性能模式，可手动配置成不清零，但不做清零操作存在泄露内核数据和用户数据的风险。

关闭大页清零的方法：

1.  登录所有参与训练的Device侧。
2.  在所有参与训练的Device侧，分别执行**echo 0 \> /proc/sys/sharepool/sharepool\_clear\_hugepage**命令即可。

    可通过调用其他接口确认是否修改成功。


