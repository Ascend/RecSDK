# 脚本工具介绍
## ckpt_convert.py 
将 mxRec生成的dense ckpt参数和转化后的sparse npy文件 转成cpu格式参数<br>
使用方法：
   进入目录：tf_serving_inference/tools<br>
   准备好 1)使用模型转化工具转化好的sparse参数npy文件，key和embedding各一个文件; 2) mxRec训练的dense ckpt参数<br>
 
   执行转化命令：
> python3 ckpt_convert.py cpu_ckpt_path mxrec_ckpt_path output_path
>>第一个参数是cpu模型的参数目录<br>
>>第二个目录是mxRec模型的参数<br>
>>第三个目录是输出目录<br>
>>注：以上模型目录一般都会带上保存的步数信息：model.ckpt-xxx，其中xxx是整数，表示在训练到多少步保存的参数<br>
>>命令执行结束后，会在output目录生成新的ckpt文件,将model.ckpt-xxx.data-00000-of-00001和model.ckpt-xxx.index文件覆盖掉cpu_ckpt_path的对应文件

## saved_mode2om.py
将saved_model格式模型转化为huawei om模型
转化OM模型:<br>
进入目录tf_serving_inference/tools<br>
更改om.sh中的--input_path路径，改为导出的saved_model路径,同时更改output_path和saved_model2om.py中的TMP_PATH<br>
执行脚本 <br>
>source /usr/local/Ascend/tfplugin/set_env.sh;sh om.sh

生成的om模型目录结构如下：
```
om_model/
├── 1
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── om_dir_1694076092
├── om_dir_1694076116
│   └── model.om
├── pb_dir_1694076092
└── pb_dir_1694076116
    └── model.pb