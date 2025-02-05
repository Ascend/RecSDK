MAX_LENGTH=50   # max length for long-sequence features, -1 means no cast
NUM_OF_PROC=10  # number of process workers, only for step6_gen_tfrecod.py 
PADDING=true   # whether generate padded dataset, if true, multi-hot fields 
                # in generated dataset will be padded to the max length, 
                # result in fiexed-length (THE DATASET WILL BE MUCH LARGER!!)

if [ "$PADDING" = "true" ]; then
    PADDING_FLAG="--padding=True"
else
    PADDING_FLAG=""
fi

echo running step1_count_vocabs.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step1_count_vocabs.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo running step2_remove_low_ids.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step2_remove_low_ids.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo running step3_map_ids.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step3_map_ids.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo running step4_split_val.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step4_split_val.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo running step5_merge_table.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step5_merge_table.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo running step6_gen_tfrecord.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step6_gen_tfrecord.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo running step7_gen_spec.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
python3 step7_gen_spec.py --length=${MAX_LENGTH} --proc=${NUM_OF_PROC} ${PADDING_FLAG}
echo process done, output will be in "aliccp_out" 
