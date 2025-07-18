# 免责声明
本样例仅供参考，部署在生产环境可能造成安全问题，建议全面测试，并自行评估潜在风险。更多详细信息请参考Tensorflow serving官方部署指导手册。

# 推理环境部署
一、安装依赖包：</p>
安装开发套件包Ascend-cann-toolkit_{version}_linux-{arch}.run</p>
安装框架插件包Ascend-cann-tfplugin_{version}_linux-{arch}.run</p>
安装其他依赖包：</p>
|依赖包   |  版本限制|
|:---|:---:|
|gcc,g++|8.4及以上版本|
|zip,unzip,libtool,automake|无特定版本要求|
|python|3.7.5|
|TensorFlow| 1.15.0|
|tensorflow-serving-api|1.15.0|
|future|无特定版本要求|
|bazel|0.24.1|
|cmake|3.14.0|
|swig|若操作系统为"aarch64"，软件安装版本需大于或等于3.0.12。若操作系统架构为"X86_64"，软件安装版本需大于或等于4.0.1|
|java|jdk-11|
|||

二、准备TF server
请根据昇腾社区提供的指导手册安装部署TF server：https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/82RC1alpha003/migration/tfmigr1/atlastfserv_26_0001.html

# 启动在线推理
1. 启动server
> 参考昇腾社区：https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/82RC1alpha003/migration/tfmigr1/atlastfserv_26_0005.html 启动TF Serving章节中的内容拉起server
2. 发起请求
> 执行脚本：sh client.sh
> 推理成功会打印请求结果

# 多流并行优化简介
背景：
> 在推荐推理场景下，模型大多存在算子数量大，算子shape小的特点，且通常会要求单次推理时间在合理范围内尽可能的提高吞吐（单次推理的数据量*单位时间内的推理次数）。<br>
> 用户在使用NPU执行推荐推理模型时，由于算子shape小，通常会出现算力利用率低从而导致性能不佳的情况。<br>

优化方法：
> 为了解决上诉问题，我们可以通过并行执行多个推理请求，在单次推理时延达标情况下极大提高吞吐。如下图所示：<br>


### 图1：
![单流执行示意图](single_stream_chart.png) 


### 图2：
![多流并行示意图](multi_stream_chart.png)


+ 图中上半部分为host，也就是cpu，下半部分是device，也就是NPU。开启多流并行时，host分多个threads下发任务（最新TFA版本支持1~16个stream，demo中默认设置了8个stream），device分多个stream执行任务（与host thread数量一致）。<br>
+ 上图中，一次推理请求有4个算子，假设单次推理数据量保持一致的情况下，device同一时间能并行处理3个算子（真实情况下，由npu调度模块根据device总核数以及每个算子需要使用的核数动态确定并行处理多少个算子）。<br>
+ 如图中时间点2，3，4，5，6，7处所示，图1中device只执行了一个算子或没有执行算子，算力利用率为33%或0%；图2中deivce同一时间执行了4个stream上的不同算子，算力利用率为100%或66%。<br>
+ 图2多流并行开启后，device在整个时间轴内执行了约18个推理请求，相比图1中device在整个时间轴内仅仅只执行了5个请求，吞吐提升到3.6倍。<br>

# 启动多流并行优化
1. 启动server
> 参考昇腾社区：https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/82RC1alpha003/migration/tfmigr1/atlastfserv_26_0005.html 启动TF Serving章节中的内容
> 将启动命令中的config文件替换为本样例中的multi_stream.cfg，随后拉起server

2. 多进程发起请求方法
> 执行脚本：sh client.sh 20 <br>
> 此处20代表着起20个client进程向server发送请求 <br>
> 此处的多线程发送请求是在模拟正式部署情况下多台机器上的client向server发送请求的情况。<br>
> 推理成功会打印请求结果 <br>

# 使用切图工具
本工具是基于cann的一个混合计算功能，开发的一个生成配置文件的工具；生成的配置文件中的in_out_pair可以控制具体下沉那些算子到npu，从而提升模型运行性能；
混合计算功能参考链接: [https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/81RC1beta1/migration/tfmigr1/tfmigr1_000074.html](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/81RC1beta1/migration/tfmigr1/tfmigr1_000074.html)
1. 进入目录：mxrec/tools/graph_partition,修改gen_config.py中的模型目录
2. 执行 python3 gen_config.py，使用生成的test1.cfg文件启动模型，使用方法如下：
> python3 gen_config.py --output_path . --tags_name serve --output_filename test1.cfg --model_path savedmodel_path<br>
+ 参数解释：output_path(输出路径),tars_name(模型tags名字多个以逗号隔开),output_filename(输出文件名),model_path(输入模型路径)<br>
+ 得到输出文件后，替换服务启动脚本中--platform_config_file参数选项即可生效
+ tag取值取决于保存模型时打的标签，当一个模型包含不同的MetaGraphDef的时候，可以通过tag来区分具体使用的MetaGraphDef，默认tag为 serve

# 性能优化
1. 具体参考optimize目录下的README.md文件