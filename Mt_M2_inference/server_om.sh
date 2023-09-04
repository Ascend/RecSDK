export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
#export PROFILING_MODE=true
#export PROFILING_OPTIONS='{"output":"/tmp/profiling","task_trace":"on","fp_point":"","aic_metrics":"PipeUtilization"}'
#/home/l00832016/M2_infer/mt_inference_model/saved_model/ \
taskset -c 0-32 tensorflow_model_server \
--model_name=saved_model \
--model_base_path=/home/l00832016/M2_infer/tools/om_model/ \
--port=9999 \
--enable_model_warmup=true \
--platform_config_file=M2.cfg \
