# tik2_custom_op

### 介绍
tik2 自定义算子入网工程


### 目录结构

```
├── CMakeLists.txt 
├── CMakePresets.json
├── README.md
├── build.sh       
├── framework                //框架适配件
├── op_host                  //算子原型&tiling
│   ├── xx.h
│   ├── xx.cc
│   ├── CMakeLists.txt
├── op_kernel                //算子kernel
│   ├── xx.cc
│   ├── CMakeLists.txt
├── cmake 
│   ├── config.cmake
│   ├── util
│       ├── makeself       
│       ├── parse_ini_to_json.py       
│       ├── gen_ops_filter.sh          
```



#### 工程约定
- 算子名称按照大驼峰命名，如 SampleTik2
- host和kernel目录下的算子文件名前缀将大驼峰转为snake,如： SampleTik2 -> sample_tik2   
- 算子tiling结构定义文件名  <snake>_tiling.h, 如： sample_tik2_tiling.h
- 算子的入口函数为算子名的snake格式，如：sample_tik2 


### 算子工程创建/添加

#### 使用msopgen工具创建/添加

需要首先定义算子的IR的json文件，下面是参考

```
[
{
        "op":"AddDSL",
        "input_desc":[
        {
                "name":"x1",
                "param_type":"required",
                "format":[
                        "NCHW"
                ],
                "type":[
                        "fp16"
                ]
        },
        {
                "name":"x2",
                "param_type":"required",
                "format":[
                        "NCHW"
                ],
                "type":[
                        "fp16"
                ]
        }
        ],
        "output_desc":[
        {
                "name":"y",
                "param_type":"required",
                "format":[
                        "NCHW"
                ],
                "type":[
                        "fp16"
                ]
        }
        ]
        "attr":[
        {
                "name":"n",
                "param_type":"required",
                "type":"int"
        }
        ]
}
]
```

2、使用msopgen命令,将上面的json文件转换为算子工程
```
#创建算子工程
msopgen gen -i xxx.json -f tf -c ai_core-ascend910 -lan cpp -m 0 -out new_tik2_custom
#算子工程里追加算子
msopgen gen -i xxx.json -f tf -c ai_core-ascend910 -lan cpp -m 1 -out ./  
```
目前追加算子有小bug，需要检查cmake/config.cmake 文件

```
set(ASCEND_COMPUTE_UNIT ascend910 ascend910b)  #检查此处的芯片类型是否涵盖了新算子的应用范围
```
3、自行补充完善算子的三个交付件

```
├── op_host                  
│   ├── xx_tiling.h                //算子tiling结构定义
│   ├── xx.cc                      //算子原型/tiling/infershape/信息库定义
├── op_kernel                
│   ├── xx.cc                      //算子kernel实现
```


#### 手工添加算子

1、将编写好的3个算子代码交付件放入工程对应目录
```
├── op_host                  
│   ├── new_op_tiling.h                //算子tiling结构定义
│   ├── new_op.cc                      //算子原型/tiling/infershape/信息库定义
├── op_kernel                
│   ├── new_op.cc                      //算子kernel实现
```
如果想删除工程里的算子， 则直接删除 op_host和op_kernel 目录下对应的算子即可。

2、检查cmake/config.cmake文件
```
set(ASCEND_COMPUTE_UNIT ascend910 ascend910b)  #检查此处的芯片类型是否涵盖了新算子的应用范围
```
3、在框架适配件中 添加算子适配件文件





### 算子包编译及安装

1.  修改CMakePresets.json 文件

将此处的ASCEND_CANN_PACKAGE_PATH 修改为正确的cann目录

```
                "ASCEND_CANN_PACKAGE_PATH": {
                    "type": "PATH",
                    "value": "/usr/local/Ascend/ascend-toolkit/latest"
                },
```

2.  设置环境变量


```
# 安装toolkit包时配置
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```

3.  编译自定义算子包

```
./build.sh
```

4. 算子包安装

执行命令，自定义算子将会自动安装到 ${ASCEND_OPP_PATH}/vendors 目录下
```
./build_out/custom_opp_<target os>_<target architecture>.run 
```



### TIK2可参考教程&代码
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/operatordevelopment/tik2opdevg/atlastik2_10_0001.html

https://codehub-y.huawei.com/TIKC-Project/tikcpp_smoke/files?ref=master&filePath=external_tik2_demo



### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


### 特技

1.  使用 Readme_XXX.md 来支持不同的语言，例如 Readme_en.md, Readme_zh.md
2.  Gitee 官方博客 [blog.gitee.com](httpsblog.gitee.com)
3.  你可以 [httpsgitee.comexplore](httpsgitee.comexplore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](httpsgitee.comgvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [httpsgitee.comhelp](httpsgitee.comhelp)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [httpsgitee.comgitee-stars](httpsgitee.comgitee-stars)
