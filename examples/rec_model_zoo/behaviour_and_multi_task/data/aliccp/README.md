# Parse Script for Dataset [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/408)

We process the dataset using the following criteria:
1. We delete the invalid data (`y`==0 && `z`==1) in datasets
2. We remove low-frequency features (feature ids that only appear once in train dataset) from the feature vocabularies, treat them as feature ids never appeared while training (`id` == 0)
3. We remap the ids of each field. After remapping, the ids of each field follow the following arrangement:
    + `id` == 0: Features that have never appeared before (including low-frequency features mentioned in step 2, and features that only appear in the test dataset)
    + `id` > 0: Legal feature, range in [1, `max_id_count`]
    + `id` < 0: illegal, should never appear
4. We randomly split the origin test dataset into test dataset and validate dataset at a ratio of 1:1
5. We reformat the datasets into `.tfrecord` format
6. **We cast all long-sequence feature, only keep specified numbers of features from the beginning**


## Scripts

We divide the parse process into seven steps:

1. Count the frequency of features each field (for long-sequence, only specified numbers from the beginning)
2. Remove low-frequency features and generate vocabularies
3. Remap ids, cast long-sequence ids
4. Split test dataset into test dataset and validate dataset
5. Merge `common_feature` table and `sample_skeleton` table, generate the final datasets in `.csv` format
6. Reformat the `.csv` dataset into `.tfrecord` 
7. Generate the specification of datasets

## Usage

1. Download the dataset from [Ali-CCP: Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/408) and unzip 

2. Organize the file structure like: 
    ```
    .
    ├── common_features_test.csv
    ├── common_features_train.csv
    ├── sample_skeleton_test.csv
    ├── sample_skeleton_train.csv
    ├── step1_count_vocabs.py
    ├── step2_remove_low_ids.py
    ├── step3_map_ids.py
    ├── step4_split_val.py
    ├── step5_merge_table.py
    ├── step6_gen_tfrecord.py
    └── step7_gen_spec.py
    ```

3. Run `run.sh`, you may need change the variables defined inside the file

    ```bash 
    # change the variables inside, then:
    bash run.sh 
    ```

4. The final output should be in `./aliccp_out`
