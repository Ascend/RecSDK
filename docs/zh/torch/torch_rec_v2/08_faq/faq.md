# FAQ<a name="ZH-CN_TOPIC_0000001579847284"></a>

## 多机训练HCCL集群通信失败<a name="ZH-CN_TOPIC_0000001795546622"></a>

**问题现象<a name="section964994911912"></a>**

HCCL集群通信失败。

**可能原因<a name="section1581817491790"></a>**

- 多机节点的NPU device ip不能互相ping通。
- 多机节点的NPU device的TLS配置不同。
- 其他

**解决方案<a name="section181445017916"></a>**

1. 检查多机节点的NPU device ip是否能互相ping通。以双机集群（节点A、B），每个节点8卡示例。
    1. 查询节点A的device ip:

        ```bash
        for i in {0..7}; do hccn_tool -i $i -ip -g ; done
        ```

    2. 在节点B上ping节点A的device ip：

        ```bash
        hccn_tool -i 0 -ping -g address 192.x.x.x
        ```

        其中192.x.x.x为节点A的rank0的device ip；0为指定使用B节点rank0 device去ping对应ip。

        若指令回显包含“0.00% packet loss”则说明能ping通；ping不通则需检查环境网络配置；

        >[!NOTE] 说明
        >若device ip配置为IPv6，查询device ip指令和ping device指令有所区别，示例：
        > - 查询device ip：
        >
        > ```bash
        > for i in {0..7}; do hccn_tool -i $i -ip -inet6 -g; done
        > ```
        >
        > - ping指定device ip：
        >
        > ```bash
        > hccn_tool -i 0 -ping -inet6 -g ipv6_address x:x:x:x
        > ```

2. 检查多机节点的NPU device的TLS配置是否相同。

    在两个节点上分别执行如下指令，查看配置是否相同：

    ```bash
    for i in {0..7}; do hccn_tool -i $i -tls -g  |grep switch; done
    ```

    若配置不同，则需修改TLS配置。

    ![](../../../figures/tf_rec_v1/zh-cn_image_0000001795706422.png)

3. 其他集合通信相关问题可参考[HCCL集合通信库](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/hccl_intro.md)的“故障诊断”章节。

## 导入dynamic_emb_extensions自定义算子包时报错<a name="ZH-CN_TOPIC_0000001809551534"></a>

**问题现象<a name="section964994911912"></a>**

使用dynamic_emb_extensions算子库调用到自定义算子时报错`ImportError: libdynamic_emb_op_npu.so: cannot open shared object file: No such file or directory`。

**可能原因<a name="section1581817491790"></a>**

自定义算子库 dynamic_emb_extensions.cpython-*.so 及其依赖的libdynamic_emb_op_*.so等动态库拷贝至Python的site-packages目录，但Linux动态链接器默认搜索路径并不包含site-packages。因此需要将site-packages路径加入LD_LIBRARY_PATH环境变量，动态链接器才能正确找到并加载所需的共享库。

**解决方案<a name="section181445017916"></a>**

将Python的site-packages路径加入LD_LIBRARY_PATH环境变量。

```bash
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('platlib'))"):$LD_LIBRARY_PATH
```
