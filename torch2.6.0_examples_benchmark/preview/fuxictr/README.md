## FuxiCTR模型库具有以下依赖项：
GitHub地址：https://github.com/reczoo/FuxiCTR/

可能出现sklearn版本过高的情况 可通过降版本至 sklearn<1.5 解决
详情见 https://github.com/reczoo/FuxiCTR/issue/156

注意本模型库源码文件中同时存在Windows(CRLF)和Unix(LF)换行符格式混用，建议在应用建议在应用patch前统一文件格式。
如果遇到patch应用失败时需注意确认涉及patch修改的对应文件的格式，否则无法正常使用

可使用'cat -A your_file.py'对文件进行确认

使用下面命令对相应的文件转换至需要的格式：
```shell 
unix2dos your_file.py #将LF转换为CRLF
dos2unix your_file.py #将CRLF转换为LF
```

## 运行环境准备
请参考：https://gitcode.com/Ascend/RecSDK/blob/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/README.md

## fuxictr模型库源码适配

进入当前目录，下载官方模型代码后，并使用patch文件进行修改。
```shell
git clone https://github.com/reczoo/FuxiCTR.git
cd FuxiCTR && git checkout b7dff736885fdb8f59387d82d08219ad2e4cae50
cp ../fuxictr_npu.patch ./ && git apply fuxictr_npu.patch
```

请通过下面指令安装其他必需的软件包。
注意需要使用 -e 在FuxiCTR目录下手动安装本地目录的fuxictr。
```shell
pip install -r requirements.txt 
pip install -e . 
```

具体模型运行脚本和可以见对应模型同名目录。

## 特殊场景
如果测试性能只推理不训练,则进入对应模型目录下，执行下面命令替换代码。
```shell
# cd FuxiCTR/model_zoo/MODEL/
sed -i 's/model.fit(train_gen, validation_data=valid_gen, \*\*params)/model.predict(valid_gen)/' run_expid.py
```
在 model.predict(valid_gen) 后代码可以全部注释，不影响profiling结果。