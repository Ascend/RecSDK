# DCNv2模型运行说明

## 代码结构
```shell
.
├── config.py  # 模型配置文件
├── delay_loss_scale.py  # loss缩放函数
├── main_mxrec.py     # 主函数
├── model.py          # DCNv2模型
├── op_impl_mode.ini  # 算子执行模式配置
├── optimizer.py      # 优化器
├── README.md         # DCNv2模型运行说明
└── run.sh            # 运行DCNv2模型的脚本
```

## 1.准备数据
参考DLRM模型中criteo_tb目录下的说明文档准备好模型所需要的数据集，放在一个目录下，比如：/data/criteo_tb/。

## 2.准备运行环境
运行环境可以参考[Rec SDK用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc2/mxRec/mxrecug/mxrecug_0007.html)
“安装部署”章节进行准备。

## 3.安装Rec SDK
Rec SDK软件包可以通过[Rec SDK用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc2/mxRec/mxrecug/mxrecug_0007.html)
“安装部署”>“环境准备”>“获取软件包”章节提供的链接进行下载，选择自己需要的架构（x86或者arm）的Rec SDK包。下载完成之后，将Rec SDK包解压，进入解压后的目录（mindxsdk-mxrec）
如下：
```shell
.
├── cust_op
│   └── cust_op_by_addr
├── examples
│   ├── DCNv2
│   ├── demo
│   └── dlrm
├── tf1_whl
│   └── mx_rec-{version}-py3-none-linux_x86_64.whl  # version为版本号
├── tf2_whl
│   └── mx_rec-{version}-py3-none-linux_x86_64.whl  # version为版本号
└── version.info
```
其中，tf1_whl和tf2_whl目录下分别是适配tf1和tf2的Rec SDK软件包，按照自己需要选择其中一个进行安装即可（用pip/pip3 install 软件包这种方式进行安装）。
确认安装Rec SDK的目录，比如Rec SDK安装在 /usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec这个目录下。

## 4.运行DCNv2模型
执行完以上步骤之后，接下来就可以运行DCNv2模型，其中run.sh就是运行的脚本，默认是8张卡。其中需要传入5个参数，分别对应：so_path、rec_package_path、hccl_cfg_json、
dlrm_criteo_data_path和ip。运行命令如：
```shell
bash run.sh {so_path} {rec_package_path} {hccl_cfg_json} {dlrm_criteo_data_path} {ip}
```
* so_path：so_path是Rec SDK中动态库的目录，一般在Rec SDK的安装目录下的libasc目录，比如：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc。
* rec_package_path：rec_package_path是Rec SDK的安装目录，比如：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec。
* hccl_cfg_json：hccl_cfg_json是hccl通信配置文件，如果配置了ip参数，这个参数就不用了，直接给一个""空字符串即可。
* dlrm_criteo_data_path：dlrm_criteo_data_path是数据集所在的目录，比如/data/criteo_tb/。
* ip：ip是运行模型的机器所在的ip，建议配置。
