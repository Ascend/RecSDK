## 适用场景说明
> 需要满足以下件的场景，适合使用本示例：
- 已经在/mxrec/cust_op/下安装完fused_sgd，希望在Tensorflow中调用该算子
- 需要完整安装mx_rec tf1安装包

## 使用方法
- 执行脚本前需要首先定义MX_REC_PACKAGE_PATH
```shell
export LD_PRELOAD=/usr/local/gcc7.3.0/lib64/libgomp.so.1
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/:$LD_LIBRARY_PATH
export MX_REC_PACKAGE_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/
```
- 利用pytest 执行脚本
```python
pytest test.py
```
- 运行结束之后会在当前目录生成汇总文件(包含精度和性能)