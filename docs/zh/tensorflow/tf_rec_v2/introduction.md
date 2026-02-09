# 简介<a name="ZH-CN_TOPIC_0000002210421025"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000002210306637"></a>

Rec SDK TensorFlow的功能涉及：

-   模型训练基础功能。支持单机单卡训练、单机多卡分布式训练，支持基于TensorFlow开发模型。
-   推荐场景特有功能。基于Rec SDK TensorFlow的稀疏表方案，Rec SDK TensorFlow提供必备功能，如特征保存和加载、特征准入、特征淘汰等。

**关键功能特性<a name="section7262101710233"></a>**

Rec SDK TensorFlow为用户提供了稀疏表创建、稀疏表查询、保存与加载、特征准入与淘汰等功能特性，用户可以在适配模型中加入想要使用的功能特性。

-   稀疏表创建

    Rec SDK Tensorflow训练框架支持稀疏表创建功能，可通过[稀疏表创建](api/model_apis.md#get_embedding_tablea-namezh-cn_topic_0000001630246521a)接口查看功能和使用实例。

-   稀疏表查询

    Rec SDK Tensorflow训练框架支持稀疏表查询功能，可通过[稀疏表查询](api/model_apis.md#embedding_lookupa-namezh-cn_topic_0000001630246521a)接口查看功能和使用实例。

-   保存与加载

    深度学习中的保存与加载是将训练好的模型参数持久化存储，并在需要时恢复使用的过程。保存通常包含模型架构、权重和优化器状态，加载则恢复模型到可用状态，实现训练中断续训或部署推理。

    Rec SDK Tensorflow训练框架支持稀疏表保存与加载功能，可通过[保存与加载](api/model_apis.md#embeddingtablesavera-namezh-cn_topic_0000001630246521a)接口查看功能和使用实例。

-   特征准入与淘汰

    当某些特征的频率过低时，对模型的训练效果不会有帮助，还会造成内存浪费以及过拟合的问题。因此需要特征准入功能来过滤掉频率过低的特征。 对于一些对训练没有帮助的特征，需要将其淘汰以免影响训练效果，同时也能节约内存。Rec SDK TensorFlow中支持特征准入与淘汰功能，可通过[特征准入与淘汰](api/model_apis.md#get_embedding_tablea-namezh-cn_topic_0000001630246521a)接口中的`min_used_times`和`max_cold_secs`参数查看说明。

## 软件架构<a name="ZH-CN_TOPIC_0000001601745068"></a>

![](../../figures/tf_rec_v1/4-mxRec架构.png)

Rec SDK TensorFlow基于推荐场景主流框架、CANN和各种硬件和网络，对于搜索、推荐、广告模型训练的应用场景需求，提供极简易用、高性能API，助力昇腾AI处理器完成搜索、推荐、广告等模型的高效训练。

**表 1**  结构图模块介绍

|Rec SDK TensorFlow模块|说明|
|--|--|
|接口层|易用性接口，简化用户接入成本，支撑用户规模化上量。|
|推荐功能层|必备核心功能，满足用户使用的要求。|
|推荐加速层|性能竞争力核心组件，为整机系统方案提供更优性能。|
|稀疏存储层|支持超10TB大规模稀疏表存储。|

## 支持的硬件和操作系统<a name="ZH-CN_TOPIC_0000001718204633"></a>

**表 2**  支持的产品列表

|产品型号|产品架构|操作系统版本|
|--|--|--|
|Atlas 800T A2 训练服务器<br>Atlas 200T A2 Box16 异构子框|<li>ARM</li><li>x86_64</li>|<li>CentOS版本：7.6</li><li>OpenEuler版本：22.03</li><li>Ubuntu版本：20.04</li>|
|Atlas 900 A3 SuperPoD 超节点|ARM|OpenEuler版本：22.03|



