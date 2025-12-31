# 附录<a name="ZH-CN_TOPIC_0000002336148833"></a>

## 用户信息列表<a name="ZH-CN_TOPIC_0000002313556722"></a>

请周期性地更新用户的密码，避免长期使用同一个密码带来的风险。

**表 1**  用户列表

|用户名|描述|初始密码|密码修改方法|
|--|--|--|--|
|root|部署Rec SDK Torch|用户自定义|使用**passwd**命令修改|
|HwHiAiUser|安装驱动，运行Demo依赖的用户|用户自定义|使用**passwd**修改|


**openeuler系统中DockerFile示例的基础镜像用户<a name="section682191514281"></a>**

|用户|初始密码|密码修改方法|
|--|--|--|
|root|无|-|
|bin|无|-|
|daemon|无|-|
|adm|无|-|
|lp|无|-|
|sync|无|-|
|shutdown|无|-|
|halt|无|-|
|mail|无|-|
|operator|无|-|
|games|无|-|
|ftp|无|-|
|nobody|无|-|
|unbound|无|-|
|tss|无|-|


**openeuler系统中RecSDK-Torch组件容器内的用户<a name="section44881041374"></a>**

|用户|描述|初始密码|密码修改方法|
|--|--|--|--|
|systemd-coredump|-|无|-|
|systemd-network|-|无|-|
|systemd-resolve|-|无|-|
|sshd|-|无|-|
|dbus|-|无|-|
|HwHiAiUser|驱动run包的运行用户|无|-|


**debian系统中DockerFile示例的基础镜像用户<a name="section16139169181617"></a>**

|用户|初始密码|密码修改方法|
|--|--|--|
|root|无|-|
|bin|无|-|
|daemon|无|-|
|lp|无|-|
|sync|无|-|
|mail|无|-|
|games|无|-|
|nobody|无|-|
|sys|无|-|
|man|无|-|
|news|无|-|
|uucp|无|-|
|proxy|无|-|
|www-data|无|-|
|backup|无|-|
|list|无|-|
|irc|无|-|
|_apt|无|-|


**debian系统中RecSDK-Torch组件容器内的用户<a name="section646063243314"></a>**

|用户|描述|初始密码|密码修改方法|
|--|--|--|--|
|systemd-network|-|无|-|
|systemd-timesync|-|无|-|
|messagebus|-|无|-|
|sshd|-|无|-|


**centos系统中DockerFile示例的基础镜像用户<a name="section378581843617"></a>**

|用户|初始密码|密码修改方法|
|--|--|--|
|root|无|-|
|bin|无|-|
|daemon|无|-|
|adm|无|-|
|lp|无|-|
|sync|无|-|
|shutdown|无|-|
|halt|无|-|
|mail|无|-|
|operator|无|-|
|games|无|-|
|ftp|无|-|
|nobody|无|-|
|systemd-network|无|-|
|dbus|无|-|


**centos系统中RecSDK-Torch组件容器内的用户<a name="section1238015439358"></a>**

|用户|描述|初始密码|密码修改方法|
|--|--|--|--|
|sshd|-|无|-|



