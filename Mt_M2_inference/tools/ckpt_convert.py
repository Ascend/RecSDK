import numpy as np
import tensorflow as tf
 
import sys
 
 
def convert(cpu_ckpt_path,npu_ckpt_path,output_path):
    reader_cpu = tf.train.NewCheckpointReader(cpu_ckpt_path)
    reader_npu = tf.train.NewCheckpointReader(npu_ckpt_path)
 
    var_map_npu = {}
    var_names_npu = reader_npu.get_variable_to_shape_map().keys()
    for var_name in var_names_npu:
        new_var = reader_npu.get_tensor(var_name)
        var_map_npu[var_name] = new_var
 
    var_names_cpu = reader_cpu.get_variable_to_shape_map().keys()
    var_map_cpu = {}
    for var_name in var_names_cpu:
        if var_name in ('embedding_table-values','embedding_table-keys'):
            print("[INFO] Hash table %s found."%var_name)
            continue
        elif var_name not in var_map_npu:
            print("[WARN] %s not found in npu ckpt!!!"%var_name)
            var_map_cpu[var_name] = tf.Variable(reader_cpu.get_tensor(var_name), name = var_name)
        else:
            print("[Info] Assign %s."%var_name)
            var_map_cpu[var_name] = tf.Variable(var_map_npu[var_name], name=var_name) 
    
    values = np.load("one_ascend_hash_embedding_embedding.npy")
    keys = np.load("one_ascend_hash_embedding_key.npy").astype(np.int32)
        
    var_map_cpu['embedding_table-values'] = tf.Variable(values,name = 'embedding_table-values')
    var_map_cpu['embedding_table-keys'] = tf.Variable(keys,name = 'embedding_table-keys')
 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, output_path)
 
    for var_name in var_names_npu:
        if var_name not in var_names_cpu:
            print("[WARN] Attention!!!! %s not found in cpu ckpt!!!"%var_name)
    print("Convert ckpt done!")
 
 
def main():
    cpu_ckpt_path = sys.argv[1]
    npu_ckpt_path = sys.argv[2]
    output_path = sys.argv[3]
    print("[INFO] cpu_ckpt_path: %s"%cpu_ckpt_path)
    print("[INFO] npu_ckpt_path: %s"%npu_ckpt_path)
    print("[INFO] output_path: %s"%output_path)
    convert(cpu_ckpt_path,npu_ckpt_path,output_path)
 
 
if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("ERROR! Usage: python3 ckpt_convert.py cpu_ckpt_path npu_ckpt_path output_path")
    else:
        main()
