# sess.run模式下demo模型运行说明

## 代码结构
```shell
.
├── config.py                 # 模型配置文件
├── dataset.py                # 生成数据集的脚本
├── demo_logger.py            # 定义logger
├── deterministic_loss        # 确定性计算loss样例
├── main.py                   # 主函数
├── model.py                  # demo模型
├── op_impl_mode.ini          # 算子执行模式配置
├── optimizer.py              # 优化器
├── random_data_generator.py  # 数据生成器
├── README.md                 # demo模型运行说明
├── run_deterministic.sh      # 运行确定性计算的脚本
├── run_mode.py               # 执行模型train、evaluate和predict的脚本
├── utils.py                  # 精度检测工具
└── run.sh                    # demo运行脚本
```

## 1.准备数据
demo样例无需从其他地方下载数据集，在demo样例中Rec SDK会自动生成数据集，详情见dataset.py和random_data_generator.py。

## 2.准备运行环境
运行环境可以参考[Rec SDK用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc3/mxRec/mxrecug/mxrecug_0004.html)
“安装部署”章节进行准备。

## 3.安装Rec SDK
Rec SDK软件包可以通过[Rec SDK用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc3/mxRec/mxrecug/mxrecug_0004.html)
“安装部署”>“环境准备”>“获取软件包”章节提供的链接进行下载，选择自己需要的架构（x86或者arm）的Rec SDK包。下载完成之后，将Rec SDK包解压，进入解压后的目录（mindxsdk-mxrec）
如下：
```shell
.
├── tf1_whl
    └── rec_sdk_common-{version}-py3-none-linux_x86_64.whl  # version为版本号
│   └── mx_rec-{version}-py3-none-linux_x86_64.whl  # version为版本号
├── tf2_whl
    └── rec_sdk_common-{version}-py3-none-linux_x86_64.whl  # version为版本号
│   └── mx_rec-{version}-py3-none-linux_x86_64.whl  # version为版本号
└── version.info
```
其中，tf1_whl和tf2_whl目录下分别是适配tf1和tf2的Rec SDK软件包，按照自己需要选择其中一个进行安装即可（用pip/pip3 install 软件包这种方式进行安装）。
确认安装Rec SDK的目录，比如Rec SDK安装在 /usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec和rec_sdk_common目录下。

## 4.运行demo模型
执行完以上步骤之后，接下来就可以运行demo模型，其中run.sh就是运行的脚本，默认是8张卡。其中需要传入ip这个参数，运行命令如：
```shell
bash run.sh main.py {ip}
```
* ip：ip是运行模型的机器所在的ip。

**Tips**：run.sh脚本中有一个参数是rec_package_path，rec_package_path是Rec SDK的安装目录，比如：/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec。
这个参数在脚本是默认的，用户需要根据自己环境中Rec SDK实际安装的路径进行配置。

## 5.开启精度对齐模式

在run.sh脚本中将 PRECISION_CHECK设置为1为开启精度对齐功能

```shell
############## 精度对齐相关 ##############
export PRECISION_CHECK=1
```

精度对齐开启后将会在run.sh同级脚本生成一个 precision_check的数据文件， 结构如下

```shell
precision_check
└── 20240807_091347
    ├── 01dump_dataset # 数据集数据
    ├── 02dump_model # dense sparse 落盘模型
    ├── 03dump_loss # 每步loss
    ├── 04dump_op # 查表 更新表相关数据
    └── dump_info.json # 模型配置信息
```

开关开启后暂只支持DUMP第1.2步数据
在默认配置下，可使用mxrec tools下的precrec-python精度工具来一键解析比较
precrec-python如何使用可以参照其中指导手册
也可自行修改代码 dump其他数据

限制（2024-0807）:
精度对齐暂不支持一表多查、准入淘汰
CACHE_MODE暂只支持 HBM DDR
其他已有功能可进行组合适配

新功能请联系开发咨询是否支持，若不支持请提需求适配