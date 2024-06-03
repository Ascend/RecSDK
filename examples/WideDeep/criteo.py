import os
import stat
import pickle
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

NAMES = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]


def make_sub_file(lines, head, src_name, sub_dir_name, sub):
    """Write sub-data.
    Args:
        :param lines: A list. Several pieces of data.
        :param head: A string. ['label', 'I1', 'I2', ...].
        :param src_name: A string. The name of data.
        :param sub_dir_name: A string.
        :param sub: A scalar(Int). Record the current number of sub file.
    :return: sub + 1.
    """
    root_path, file_path = os.path.split(src_name)
    file_name, suffix = file_path.split('.')
    split_file_name = file_name + "_" + str(sub).zfill(2) + "." + suffix
    split_file = os.path.join(root_path, sub_dir_name, split_file_name)
    if not os.path.exists(os.path.join(root_path, sub_dir_name)):
        os.mkdir(os.path.join(root_path, sub_dir_name))

    modes = stat.S_IWUSR | stat.S_IRUSR
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    f = os.fdopen(os.open(split_file, flags, modes), 'w')
    try:
        f.writelines([head])
        f.writelines(lines)
        return sub + 1
    finally:
        f.close()


def split_byline_count(filename, count, sub_dir_name):
    """Split File.
    Note: You can specify how many rows of data each sub file contains.
    Args:
        :param filename: A string.
        :param count: A scalar(int).
        :param sub_dir_name: A string.
    :return:
    """
    f = open(filename, 'r')
    try:
        head = f.readline()
        buf = []
        sub = 1
        for line in f:
            buf.append(line)
            if len(buf) == count:
                sub = make_sub_file(buf, head, filename, sub_dir_name, sub)
                buf = []
        if len(buf) != 0:
            make_sub_file(buf, head, filename, sub_dir_name, sub)
    finally:
        f.close()


def get_split_file_path(parent_path=None, dataset_path=None, sample_num=4600000):
    """Get the list of split file path.
    Note: Either parent_path or dataset_path must be valid.
    If exists dataset_path + "/split", parent_path = dataset_path + "/split".
    Args:
        :param parent_path: A string. split file's parent path.
        :param dataset_path: A string.
        :param sample_num: A int. The sample number of every split file.
    :return: A list. [file1_path, file2_path, ...]
    """
    sub_dir_name = 'split'
    if parent_path is None and dataset_path is None:
        raise ValueError('Please give parent path or file path.')
    if parent_path is None and os.path.exists(os.path.join(os.path.dirname(dataset_path), sub_dir_name)):
        parent_path = os.path.join(os.path.dirname(dataset_path), sub_dir_name)
    elif parent_path is None or not os.path.exists(parent_path):
        split_byline_count(dataset_path, sample_num, sub_dir_name)
        parent_path = os.path.join(os.path.dirname(dataset_path), sub_dir_name)
    split_file_name = os.listdir(parent_path)
    split_file_name.sort()
    split_file_list = [parent_path + "/" + file_name for file_name in split_file_name if file_name[-3:] == 'txt']
    return split_file_list


def get_fea_map(fea_map_path=None, split_file_list=None):
    """Get feature map.
    Note: Either parent_path or dataset_path must be valid.
    If exists dir(split_file_list[0]) + "/fea_map.pkl", fea_map_path is valid.
    If fea_map_path is None and you want to build the feature map,
    the default file path is the parent directory of split file + "fea_map.pkl".
    Args:
        :param fea_map_path: A string.
        :param split_file_list: A list. [file1_path, file2_path, ...]
    :return: A dict. {'C1':{}, 'C2':{}, ...}
    """
    if fea_map_path is None and split_file_list is None:
        raise ValueError('Please give feature map path or split file list.')
    if fea_map_path is None and split_file_list is not None:
        fea_map_path = os.path.join(os.path.dirname(split_file_list[0]), "fea_map.pkl")
    if os.path.exists(fea_map_path) and fea_map_path[-3:] == 'pkl':
        with open(fea_map_path, 'rb') as f:
            fea_map = pickle.load(f)
        return fea_map
    fea_map = {}
    for file_open in tqdm(split_file_list):
        f = open(file_open)
        for line in f:
            row = line.strip('\n').split('\t')
            for i in range(14, 40):
                if row[i] == '':
                    continue
                name = NAMES[i]
                fea_map.setdefault(name, {})
                if fea_map[name].get(row[i]) is None:
                    fea_map[name][row[i]] = len(fea_map[name])
            for j in range(1, 14):
                if row[j] == '':
                    continue
                name = NAMES[j]
                fea_map.setdefault(name, {})
                fea_map[name].setdefault('min', float(row[j]))
                fea_map[name].setdefault('max', float(row[j]))
                fea_map[name]['min'] = min(fea_map[name]['min'], float(row[j]))
                fea_map[name]['max'] = max(fea_map[name]['max'], float(row[j]))
        f.close()
    for i in range(14, 40):
        fea_map[NAMES[i]]['-1'] = len(fea_map[NAMES[i]])
    fea_map_path = os.path.join(os.path.dirname(split_file_list[0]), "fea_map.pkl")


    modes = stat.S_IWUSR | stat.S_IRUSR
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    with os.fdopen(os.open(fea_map_path, flags, modes), 'wb') as fd:
        pickle.dump(fea_map, fd, pickle.HIGHEST_PROTOCOL)

    return fea_map


def rec_kbins_discretizer(dat, n_bins, min_max_dict):
    """Bin continuous data into intervals.
    Note: The strategy is "uniform".
    Args:
        :param dat: A dataframe.
        :param n_bins: A scalar(int).
        :param min_max_dict: A dict such as {'min': , 'max': }.
    :return: The new  dataframe.
    """
    features = dat.columns
    n_features = len(features)
    bin_edges = np.zeros(n_features, dtype=object)
    for idx, feature in enumerate(features):
        bin_edges[idx] = np.linspace(min_max_dict[feature]['min'], min_max_dict[feature]['max'], n_bins + 1)
        rtol = 1.e-5
        atol = 1.e-8
        eps = atol + rtol * np.abs(dat[feature])
        dat[feature] = np.digitize(dat[feature] + eps, bin_edges[idx][1:])
    return dat


def convert_input2tfrd(in_file_path, out_file_path):
    """
    txt to tfrecords
    """
    def make_example(label_list, dense_feat_list, sparse_feat_list):
        # '1.0' >> 1.0 >> 1
        dense_feature = np.array(np.array(dense_feat_list, dtype=np.float32), dtype=np.int64).reshape(-1)
        sparse_feature = np.array(np.array(sparse_feat_list, dtype=np.float32), dtype=np.int64).reshape(-1)
        label = np.array(np.array(label_list, dtype=np.float32), dtype=np.int64).reshape(-1)
        feature_dict = {"dense_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=dense_feature)),
                        "sparse_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_feature)),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return example

    file_name = out_file_path + in_file_path[-12:-4] + '.tfrecords'
    file_writer = tf.io.TFRecordWriter(file_name)

    with open(in_file_path, encoding='utf-8') as file_in:

        for i, line in tqdm(enumerate(file_in)):

            line = line.strip('\n')
            items = line.split('\t')
            if len(items) != 40:
                continue
            label = int(items[0])
            dense = items[1:14]
            sparse = items[14:]

            ex = make_example(label, dense, sparse)
            serialized = ex.SerializeToString()
            file_writer.write(serialized)

        file_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get datasets')
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')

    args, _ = parser.parse_known_args()
    data_path = args.data_path
    output_path = args.output_path

    # get txt_list
    file_split_list = get_split_file_path(dataset_path=data_path)
    # get feature_map
    feature_map = get_fea_map(split_file_list=file_split_list)

    for file in tqdm(file_split_list):

        # read data
        data_df = pd.read_csv(file, sep='\t', header=None, names=NAMES)
        # name feature
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]
        # data processing
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)
        # sparse feature: mapping
        for col in sparse_features:
            data_df[col] = data_df[col].map(lambda x: feature_map[col][x])
        # dense feature: Bin continuous data into intervals.
        data_df[dense_features] = rec_kbins_discretizer(data_df[dense_features], 1000, feature_map)
        # add offsets
        slot_size_array = [1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001,
                           1462, 585, 10131228, 2202609, 307, 25, 12519, 635, 5, 93147, 5685, 8351594, 3196,
                           29, 14994, 5461307, 12, 5654, 2174, 5, 7046548, 19, 17, 286182, 106, 142573]
        offset_size_list = np.cumsum([0] + slot_size_array[:-1])
        for col_index in range(1, len(offset_size_list) + 1):
            data_df.iloc[:, col_index] += offset_size_list[col_index - 1]
        # save to txt
        data_df.to_csv(file, sep='\t', index=False, header=False)
        # txt to tfrecords
        convert_input2tfrd(in_file_path=file, out_file_path=output_path)





