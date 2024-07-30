# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import stat
import pickle
import shutil
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf

# All column names of census dataset
COLUMN_NAMES = [
        'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
        'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
        'stock_dividends', 'tax_filer_stat', 'region_prev_res',
        'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ', 'instance_weight',
        'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
        'mig_prev_sunbelt', 'num_emp', 'fam_under_18', 'country_father',
        'country_mother', 'country_self', 'citizenship', 'own_or_self',
        'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k'
]

# Sparse feature column names 
CATEGORICAL_COLUMNS = [
        'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'hs_college', 'major_ind_code', 'major_occ_code', 'race',
        'hisp_origin', 'sex', 'union_member', 'unemp_reason',
        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res',
        'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ', 'mig_chg_msa',
        'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
        'fam_under_18', 'country_father', 'country_mother', 'country_self',
        'citizenship', 'vet_question'
]

DENSE_COLUMNS = [ 
    'age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'stock_dividends', 'instance_weight',
    'num_emp', 'own_or_self', 'vet_benefits', 'weeks_worked', 'year'
]

LABEL_COLUMNS = ['income_50k', 'marital_stat']


def dataframe_column_unique(row_dataframe):
    # Get the unique value of each column
    unique_counts = row_dataframe.nunique(axis=0)
    
    #Filter out the columns that need to be deduplicated
    cols_to_drop_duplicates = row_dataframe.columns[unique_counts < len(row_dataframe)]
    
    df_unique = row_dataframe.drop_duplicates(subset=cols_to_drop_duplicates)
    
    return df_unique
    

def fun1(x):
    if x == ' 50000+.':
        return 1
    else:
        return 0


def fun2(x):
    if x == ' Never married':
        return 1
    else:
        return 0


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
    for file_open in split_file_list:
        fea_dataframe = pd.read_csv(file_open, names=COLUMN_NAMES, header=None)
        fea_unique_dataframe = dataframe_column_unique(fea_dataframe)
        
        for fea_column in CATEGORICAL_COLUMNS:
            
            for fea_value in fea_unique_dataframe[fea_column].to_list():
                fea_map.setdefault(fea_column, {})
                if fea_map.get(fea_column).get(fea_value) is None:
                    fea_map.get(fea_column).update({fea_value: len(fea_map.get(fea_column))})
            
    fea_map_path = os.path.join(os.path.dirname(split_file_list[0]), "fea_map.pkl")

    modes = stat.S_IWUSR | stat.S_IRUSR
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    with os.fdopen(os.open(fea_map_path, flags, modes), 'wb') as fd:
        pickle.dump(fea_map, fd, pickle.HIGHEST_PROTOCOL)

    return fea_map


def convert_input2tfrd(data_frame, in_file_path, out_file_path):
    """
    txt to tfrecords
    """
    
    def make_example(label_list, dense_feat_list, sparse_feat_list):
        dense_feature = np.array(dense_feat_list, dtype=np.float).reshape(-1)
        sparse_feature = np.array(sparse_feat_list, dtype=np.int64).reshape(-1)
        label = np.array(label_list, dtype=np.int64).reshape(-1)
        feature_dict = {
                    "dense_feature": tf.train.Feature(float_list=tf.train.FloatList(value=dense_feature)),
                    "sparse_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_feature)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return example

    file_name = os.path.join(out_file_path, os.path.basename(in_file_path) + '.tfrecord')
    file_writer = tf.io.TFRecordWriter(file_name)

    for _, row_info in data_frame.iterrows():
        labels = row_info[LABEL_COLUMNS].values
        dense = row_info[DENSE_COLUMNS].values
        sparse = row_info[CATEGORICAL_COLUMNS].values
        
        ex = make_example(label_list=labels, dense_feat_list=dense, sparse_feat_list=sparse)
        
        serialized = ex.SerializeToString()
        file_writer.write(serialized)

    file_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get datasets')
    parser.add_argument('--train_data_path')
    parser.add_argument('--test_data_path')
    parser.add_argument('--output_path')

    args, _ = parser.parse_known_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    output_path = args.output_path
        
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # get txt_list
    file_path_dict = {'train': train_data_path, 'test': test_data_path}
    # get feature_map
    feature_map = get_fea_map(split_file_list=list(file_path_dict.values()))

    for class_usage, file_path in file_path_dict.items():

        # read data
        data_df = pd.read_csv(file_path, sep=',', header=None, names=COLUMN_NAMES)

        # data processing
        data_df[DENSE_COLUMNS] = data_df[DENSE_COLUMNS].fillna(0)
        # sparse feature: mapping
        for col in CATEGORICAL_COLUMNS:
            try:
                data_df[col] = data_df[col].map(lambda x: feature_map[col][x])
            except KeyError as er:
                raise KeyError("Feature {} not found in dataset".format(col)) from er

        data_df[LABEL_COLUMNS[0]] = data_df[LABEL_COLUMNS[0]].apply(
            lambda x: fun1(x))

        data_df[LABEL_COLUMNS[1]] = data_df[
            LABEL_COLUMNS[1]].apply(lambda x: fun2(x))

        # add offsets
        slot_size_array = []
        for i in CATEGORICAL_COLUMNS:
            if feature_map.get(i) is not None:
                slot_size_array.append(len(feature_map.get(i)))
            else:
                slot_size_array.append[0]
        
        offset_size_list = np.cumsum([0] + slot_size_array[:-1])

        for ind_, slot_column in enumerate(CATEGORICAL_COLUMNS):
            data_df[slot_column] += offset_size_list[ind_]
        
        output_path_ = os.path.join(output_path, class_usage)
        os.makedirs(output_path_, exist_ok=True)

        # txt to tfrecords
        convert_input2tfrd(data_frame=data_df, in_file_path=file_path, out_file_path=output_path_)