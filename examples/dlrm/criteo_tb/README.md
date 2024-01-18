# 准备criteo_TB原始数据
首先从[官网](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
下载24天的原始数据集，执行命令如下。所有文件保存下来大概需要365G。

`curl -O https://storage.googleapis.com/criteo-cail-datasets/day_{`seq -s “,” 0 23`}.gz`

然后将下载好的24个文件解压，解压后的文件需要占用1035G。

# 原始数据集转tfrecord
运行转换脚本：

`python3.7 gen_ttf.py --train_data_dir train_dir --test_data_dir test_dir --tf_base_dir save_base_dir`

参数说明：

- train_data_dir: 解压后训练集路径，该路径下存放day_0,day_1,...,day_22
- test_data_dir: 解压后测试集路径，该路径下存放day_23
- tf_base_dir：tfrecord存放路径，磁盘至少需要633G

