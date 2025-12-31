# 常用操作<a name="ZH-CN_TOPIC_0000001630246513"></a>

## （可选）片上内存侧动态扩容算子包安装<a name="ZH-CN_TOPIC_0000001630046409"></a>

如需使用片上内存侧动态扩容功能，请参考本步骤编译安装片上内存侧动态扩容算子包。

**操作步骤<a name="section15154203811598"></a>**

1.  执行如下命令，在容器中设置CANN相关环境变量。

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

2.  下载[Rec SDK](https://gitcode.com/Ascend/RecSDK/tree/branch_v7.2.0-RC1/cust_op)。
3.  解压压缩包，进入“cust\_op/ascendc\_op/ai\_core\_op/cust\_op\_by\_addr“路径下，参考以下命令编译并安装片上内存侧动态扩容算子包。

    ```bash
    bash run.sh
    ```


## 查看Rec SDK TensorFlow安装与卸载记录<a name="ZH-CN_TOPIC_0000001683896117"></a>

Rec SDK TensorFlow为Wheel包格式，安装、卸载日志记录在系统history中。

**查看安装、卸载的历史记录<a name="section1220492120526"></a>**

当注销系统或者退出容器（Rec SDK TensorFlow通常是在容器中安装、运行）时会将history中的历史命令记录保存到“\~/.bash\_history”文件中。所以，可以直接查看“.bash\_history”文件就能找到Rec SDK TensorFlow安装、卸载记录。

**修改历史记录的保存数量<a name="section56389529527"></a>**

在Linux系统中，history命令一般默认保存最新的1000条命令。如果需要修改保存的命令数量，比如只保留200条历史命令，则可以在“/etc/profile“文件中修改HISTSIZE环境变量。修改方法如下：

-   使用编辑器（如vim编辑器）修改。
-   使用sed直接修改，命令如下：

    **sed -i 's/^HISTSIZE=**_number_**/HISTSIZE=**_newNumber_**/' /etc/profile**，*number*表示修改前的命令数量，*newNumber*表示修改后的命令数量。以保存的命令数量从1000改为200为例：

    ```bash
    sed -i 's/^HISTSIZE=1000/HISTSIZE=200/' /etc/profile
    ```

修改完成之后需要执行**source /etc/profile**使环境变量生效。

**修改历史命令文件时间戳<a name="section18178420544"></a>**

如果需要在历史命令文件中有时间戳记录，可以在“/etc/profile“中添加如下配置：

**HISTTIMEFORMAT='%F %T '**

添加完成之后需要执行**source /etc/profile**命令使环境变量生效。添加时间戳之后，history命令结果如图所示：

```bash
2023-08-18 10:01:57 pip3 install mx_rec-5.0.T104-py3-none-linux_x86_64.whl --force-reinstall
2023-08-18 10:01:57 pip3 install mx_rec-5.0.T104-py3-none-linux_x86_64.whl --force-reinstall
2023-08-18 10:04:37 history | grep "pip3 install"
2023-08-18 10:10:17 history | grep "pip3 install"
```

此外，如果需要将历史命令记录在自定义文件中，可以在“/etc/profile“中设置HISTFILE环境变量，设置完成之后执行**source /etc/profile**命令使环境变量生效。比如：

```bash
HISTDIR=~/log/RecSDK   # 配置历史命令记录保存文件
HISTFILE="$HISTDIR/RecSDK.log"
mkdir -p $HISTDIR
chmod 750 $HISTDIR
touch $HISTFILE
chmod 640 $HISTFILE
USER_IP=`who -u am i 2>/dev/null| awk '{print $NF}'|sed -e 's/[()]//g'`
if [ -z $USER_IP ]
then
  USER_IP=`hostname`
fi
export HISTTIMEFORMAT="%F %T $USER_IP:`whoami` "    # history命令显示格式：时间、IP、用户名、执行命令 
PROMPT_COMMAND=' { date "+%Y-%m-%d %T - $(history 1 | { read x cmd; echo "$cmd"; })"; } >> $HISTFILE'    # 实时将history命令写到配置的文件里
```

其中日志文件路径为“\~/log/RecSDK”，请保证磁盘空间足够，日志文件设置权限为640。


