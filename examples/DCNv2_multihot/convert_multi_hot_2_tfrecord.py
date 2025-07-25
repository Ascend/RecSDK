#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import argparse
import zipfile
import struct

import numpy as np
import tensorflow as tf
from numpy.linalg.linalg import fortran_int
from tensorflow_core import shape

CAT_FEATURE_COUNT = 26
NUM_EMBEDDINGS_PER_FEATURE = [40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000, 3067956, 405282, 10,
                              2209, 11938, 155, 4, 976, 14, 40000000, 40000000, 40000000, 590152, 12973, 108, 36]
MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
ROWS_PER_FILE = 1024 * 1000
LINER_PER_SAMPLE = 1024


def parse_arguments():
    parser = argparse.ArgumentParser(description="data trans to tfrecord")

    parser.add_argument("--npz_path", type=str, required=True, help="npz file data path")
    parser.add_argument("--npy_path", type=str, required=True, help="npy file data path")
    parser.add_argument("--output_path", type=str, required=True, help="final tfrecord file output path")

    return parser.parse_args()


def mkdir_path(file_path: str):
    if not os.path.exists(file_path):
        dir_mode = 0o750
        os.makedirs(file_path, dir_mode)


def _load_from_npz(fname: str, npz_name: str, row_num: int, feature_index: int):
    """
    Load a numpy array from a .npz file using memory mapping.

    Args:
        fname (str): Path to the .npz file.
        npz_name (str): Name of the array to load from the .npz file.

    Returns:
        np.memmap: Memory-mapped array from the specified .npz file.
    """
    try:
        with zipfile.ZipFile(fname) as zf:
            info = zf.NameToInfo[npz_name]

            if info.compress_type != 0:
                raise ValueError("Expected uncompressed data.")

            zf.fp.seek(info.header_offset + len(info.FileHeader()) + 20)

            zf.open(npz_name, 'r')
            version_data = np.lib.format.read_magic(zf.fp)
            if version_data == (1, 0):
                npz_shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(zf.fp)
            elif version_data == (2, 0):
                npz_shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(zf.fp)
            off_set = zf.fp.tell()

            # Create memory map
            return np.memmap(zf.filename, dtype=dtype, shape=npz_shape, order="F" if fortran_order else "C", mode="r",
                             offset=off_set)

    except FileNotFoundError as fe:
        raise FileNotFoundError(f"The file {fname} was not found.") from fe
    except KeyError as ke:
        raise KeyError(f"The array {npz_name} was not found in the file.") from ke
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}") from e


def make_example(label_fes, dense_fes, sparse_fes):
    label_fes = np.array(label_fes, dtype=np.int64).reshape(-1)
    dense_fes = np.array(dense_fes, dtype=np.int64).reshape(-1)

    feature_dict = {"dense_feature": tf.train.Feature(float_list=tf.train.FloatList(value=dense_fes)),
                    "sparse_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_fes)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label_fes))
                    }

    example_ = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_


def _get_sparse_features(index: int, sparse_npys: list) -> np.ndarray:
    """
    Extract sparse features from the given numpy arrays based on the specified index.

    Args:
        index (int): The index of the sample to extract features for.
        sparse_npys (list of np.ndarray): List of numpy arrays containing sparse features.

    Returns:
        np.ndarray: A flattened array of concatenated sparse features for the specified index.
    """
    sparse_features_list = []

    # Iterate through each feature array
    for feat_index, feat_array in enumerate(sparse_npys):
        # Extract the relevant portion of the feature array for the current index
        feat_slice = feat_array[index * LINER_PER_SAMPLE:(index + 1) * LINER_PER_SAMPLE]

        # Calculate the new feature information by adding the offset based on the feature index
        feat_offset = feat_slice[:] + sum(NUM_EMBEDDINGS_PER_FEATURE[0:feat_index])

        # Append the processed feature information to the list
        sparse_features_list.append(feat_offset)

    # Concatenate all sparse features along the first axis, convert to int64, and flatten the result
    return np.concatenate(sparse_features_list, axis=1, dtype=np.int64).reshape(-1)

if __name__ == '__main__':
    args = parse_arguments()

    sparse_data_source_path = args.npz_path
    dense_label_data_path = args.npy_path
    output_path = args.output_path

    mkdir_path(output_path)

    for day_n in range(0, 24, 1):

        sparse_file_name = 'day_{}_sparse_multi_hot.npz'.format(day_n)
        dense_file_name = 'day_{}_dense.npy'.format(day_n)
        label_file_name = 'day_{}_labels.npy'.format(day_n)

        part_file_index = 0

        file_name = os.path.join(output_path, "{}".format(day_n) + "_part_{:0>8d}.tfrecord")

        sparse_list = []
        sparse_npz_path = os.path.join(sparse_data_source_path, sparse_file_name)
        sparse_npy_list = []

        dense = np.load(os.path.join(dense_label_data_path, dense_file_name)).reshape(-1)
        label = np.load(os.path.join(dense_label_data_path, label_file_name)).reshape(-1)

        file_writer = tf.python_io.TFRecordWriter(file_name.format(part_file_index))

        all_rows = len(label)
        for feat_id_num in range(CAT_FEATURE_COUNT):
            sparse_npy_list.append(_load_from_npz(sparse_npz_path, f"{feat_id_num}.npy", all_rows, feat_id_num))

        DENSE_FEATURE_LEN = 13 * LINER_PER_SAMPLE

        current_rows = 0

        for line_index in range(all_rows // LINER_PER_SAMPLE):
            label_features = label[line_index * LINER_PER_SAMPLE:(line_index + 1) * LINER_PER_SAMPLE]
            dense_features = dense[line_index * DENSE_FEATURE_LEN:(line_index + 1) * DENSE_FEATURE_LEN]
            sparse_features = _get_sparse_features(line_index, sparse_npy_list)

            if len(sparse_features) != 1024 * 214:
                raise ValueError("Sparse feature size is error, current size: {}".format(len(sparse_features)))

            ex = make_example(label_features, dense_features, sparse_features)

            file_writer.write(ex.SerializeToString())

            current_rows += LINER_PER_SAMPLE

            if current_rows % ROWS_PER_FILE == 0:

                part_file_index += 1
                file_writer.close()
                file_writer = tf.python_io.TFRecordWriter(file_name.format(part_file_index))
        os.remove(sparse_npz_path)
