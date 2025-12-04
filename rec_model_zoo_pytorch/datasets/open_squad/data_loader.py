# Copyright 2023 ByteDance and/or its affiliates.
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
# To support feature cache.
import pickle
import collections
import logging

from tqdm import tqdm
import numpy as np

from datasets.data_loader import Dataset
from datasets.open_squad.create_squad_data import read_squad_examples, convert_examples_to_features
from transformers import BertTokenizer, AutoTokenizer

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT32": np.float32,
    "LONG": int,
    "INT32": np.int32,
    "INT64": np.int64
}

MAX_SEQ_LENGTH = 384
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128

log = logging.getLogger("SQUAD")


class SquadLoader(Dataset):
    def __init__(self, config):
        super(SquadLoader, self).__init__(config)

        log.info("Initial...")
        self.config = config
        model = self.config["model"]
        total_count_override = None
        perf_count_override = None
        eval_features = []
        self.batched_data = []
        # Load features if cached, convert from examples otherwise.
        input_file = "{}/dev-v1.1.json".format(config['dataset_path'])
        cache_path = '{}/eval_features_'.format(config['dataset_path']) + self.config[
            'model'] + '.pickle'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                eval_features = pickle.load(cache_file)
            eval_examples = read_squad_examples(input_file=input_file,
                                                is_training=False,
                                                version_2_with_negative=False)
        else:
            tokenizer = self.get_tokenizer()
            eval_examples = read_squad_examples(input_file=input_file,
                                                is_training=False,
                                                version_2_with_negative=False)

            def append_feature(feature):
                eval_features.append(feature)

            convert_examples_to_features(examples=eval_examples,
                                         tokenizer=tokenizer,
                                         max_seq_length=MAX_SEQ_LENGTH,
                                         doc_stride=DOC_STRIDE,
                                         max_query_length=MAX_QUERY_LENGTH,
                                         is_training=False,
                                         output_fn=append_feature,
                                         verbose_logging=False)

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(eval_features, cache_file)

        self.eval_features = eval_features
        self.eval_examples = eval_examples
        self.count = total_count_override or len(self.eval_features)
        self.items = len(self.eval_features)
        self.perf_count = perf_count_override or self.count
        self.model = model
        self.cur_bs = 1
        self.batch_num = int(self.items / self.cur_bs)

        # save mask name to help setting the the results at unmasked positions to zero
        if "roberta" in self.model or "torch" in self.model:
            self.mask_name = "attention_mask.1"
        else:
            self.mask_name = "input_mask:0"
    
    def get_tokenizer(self):
        log.info("Start to generate data")
        if "roberta" in self.config['model']:
            tokenizer = AutoTokenizer.from_pretrained(
                "csarron/roberta-base-squad-v1")
        elif "albert" in self.config['model']:
            tokenizer = AutoTokenizer.from_pretrained(
                "madlag/albert-base-v2-squad")
        elif "deberta" in self.config['model']:
            tokenizer = AutoTokenizer.from_pretrained(
                "Palak/microsoft_deberta-base_squad")
        else:
            tokenizer = BertTokenizer(
                "{}/vocab.txt".format(self.config['dataset_path']))
        return tokenizer

    def name(self):
        return self.config['dataset_name']

    def preprocess(self):
        log.info("Preprocessing...")

        self.rebatch(self.cur_bs, skip=False)

    def get_token_type_ids(self, features, j):
        if "roberta" in self.model:
            features['token_type_ids.1'].append(
                np.zeros((384,)))
        elif "deberta" in self.model:
            features['token_type_ids'].append(
                self.eval_features[j].segment_ids)
        else:
            features['token_type_ids.1'].append(
                self.eval_features[j].segment_ids)
        return features

    def rebatch(self, new_bs, skip=True):
        log.info("Rebatching batch size to: {} ...".format(new_bs))

        if self.cur_bs == new_bs and skip:
            return

        self.cur_bs = new_bs
        self.batch_num = int(self.items / self.cur_bs)
        for i in tqdm(range(self.batch_num)):
            features = collections.defaultdict(list)
            for j in range(i * self.cur_bs, (i + 1) * self.cur_bs):
                if "torch" in self.model:
                    features['input_ids.1'].append(
                        self.eval_features[j].input_ids)
                    features['attention_mask.1'].append(
                        self.eval_features[j].input_mask)
                    features = self.get_token_type_ids(features, j)
                else:
                    features['input_ids:0'].append(
                        self.eval_features[j].input_ids)
                    features['input_mask:0'].append(
                        self.eval_features[j].input_mask)
                    features['segment_ids:0'].append(
                        self.eval_features[j].segment_ids)
            self.batched_data.append(features)

    def get_samples(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID is out of range")
        return self.batched_data[sample_id], []

    def get_id(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID is out of range")
        return [
            self.eval_features[i].unique_id
            for i in range(sample_id * self.cur_bs, (sample_id + 1) *
                           self.cur_bs)
        ]
