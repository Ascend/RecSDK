#### 脚本用途

该脚本用于统计mxRec在host侧运行时的各种日志信息， 并输出对应结果

#### 使用注意：

1. 使用需要在mxRec对应训练脚本中配置系统环境变量 STAT_ON, 0为关闭统计，1为开启统计 mxRec默认关闭统计；
2. 请把训练日志重定向到固定目录；
3. 请配置脚本中几个参数的具体值 
   TARGET_REC_LOG_PATH 为mxrec产生的训练日志路径
   RUN_MODE 统计的运行模式，现支持的模式已在脚本中列出
   RANK_SIZE 训练脚本运行的rank size既几卡运行
   DISPLAY_MODE 统计信息展示的方式，支持落盘以及打屏 分别是 "save_log" or "print_screen"
   DISPLAY_INTERVAL 统计日志的间隔， 既多少步输出一次统计信息
4. 脚本可随训练进程同步统计，训练结束后请手动停止脚本释放资源
5. 请确保rec日志等于或低于为info