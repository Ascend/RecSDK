# coding: utf-8
import argparse
import json

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, required=True, help='path of the dataset')


def static_key_count(file_path):
    admit_threshold = 30 
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.batch(int(1), drop_remainder=False)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    offset_value = 2**48
    shift = 1
    result = {} 
    table_list = ["history_poi_seq_id_list#vector", "poi_id_end2end#vector", "rt_day_click_event_poi_id_list"]
    with tf.Session() as sess:
        while True:
            try:
                examples = sess.run(next_element)
                example = tf.train.Example.FromString(examples[0])
                features = example.features
                feature = features.feature

                for name, values in feature.items():
                    num_list = []
                    if name in table_list:
                        num_list = values.int64_list.value
                    if name not in table_list:
                        continue

                    if len(num_list) == 0:
                        print("===================")
                        num_list = [0]

                    for num in num_list:
                        num = num % offset_value + shift * offset_value
                        result[num] = result.get(num, 0) + 1

            except tf.errors.OutOfRangeError:
                print("EOS: OutOfRangeError")
                break
    temp = {}
    for key, value in result.items():
        if value >= admit_threshold:
            temp[key] = value
    sorted_result = dict(sorted(temp.items(), key=lambda x: x[1], reverse=True))
    with open("key_count30.json", "w") as f:
        json.dump(sorted_result, f, indent=4)

    print(sorted_result)


if __name__ == "__main__":
    args = parser.parse_args()
    static_key_count(args.file_path)

