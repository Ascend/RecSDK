#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=1
#export PROFILING_MODE=true
#export PROFILING_OPTIONS='{"output":"/tmp/profiling","task_trace":"on","fp_point":"","aic_metrics":"PipeUtilization"}'
taskset -c 0-32 tensorflow_model_server \
--model_name=saved_model \
--model_base_path=/home/l00832016/inference/mt_inference_model/saved_model/ \
--port=9999 \
--enable_model_warmup=true \
--platform_config_file=inference.cfg \

