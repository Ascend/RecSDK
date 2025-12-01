# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
from multiprocessing import Process, Queue
from collections import defaultdict

import numpy as np

from torch.utils.data import Dataset, DataLoader

np.random.seed(2024)


class Ml1mDataset(Dataset):
    def __init__(self, params, dataset, usernum, itemnum):
        self.params = params
        self.dataset = dataset
        self.usernum = usernum
        self.itemnum = itemnum
        self.length = usernum // params.batch_size
        # global dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_features = process_one_batch_data(self.params, self.dataset, self.usernum, self.itemnum)
        return batch_features, None


def generate_dataloader(dataset):
    return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], drop_last=True)


def load_data(params):
    dataset = data_partition(params.dataset)
    [user_train, user_valid, user_test, user_num, item_num] = dataset

    train_dataset = Ml1mDataset(params, user_train, user_num, item_num)
    test_dataset = Ml1mDataset(params, user_test, user_num, item_num)
    val_dataset = Ml1mDataset(params, user_valid, user_num, item_num)
    # batch_sizeand num_worker由内部处理
    train_loader = generate_dataloader(train_dataset)
    test_loader = generate_dataloader(test_dataset)
    val_loader = generate_dataloader(val_dataset)
    spec = {"user_num": user_num, "item_num": item_num}
    return train_loader, user_valid, user_test, spec


def process_one_batch_data(args, raw_data, usernum, itemnum):
    sampler = WarpSampler(raw_data, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=10)
    u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
    u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
    features = {"seq": seq, "pos": pos, "neg": neg}
    return features


# sampler for batch generation
def random_neq(left, right, sequence):
    tgt = np.random.randint(left, right)
    while tgt in sequence:
        tgt = np.random.randint(left, right)
    return tgt


class ML1MSampler:
    def __init__(self, user_train, user_num, item_num, batch_size, maxlen):
        self.user_train = user_train
        self.user_num = user_num
        self.item_num = item_num
        self.batch_size = batch_size
        self.maxlen = maxlen

    def sample_function(self, result_queue):
        def sample(uid):

            # uid = np.random.randint(1, usernum + 1)
            while len(self.user_train[uid]) <= 1:
                uid = np.random.randint(1, self.user_num + 1)

            seq = np.zeros([self.maxlen], dtype=np.int32)
            pos = np.zeros([self.maxlen], dtype=np.int32)
            neg = np.zeros([self.maxlen], dtype=np.int32)
            nxt = self.user_train[uid][-1]
            idx = self.maxlen - 1

            ts = set(self.user_train[uid])
            for i in reversed(self.user_train[uid][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = random_neq(1, self.item_num + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1:
                    break

            return (uid, seq, pos, neg)

        uids = np.arange(1, self.user_num + 1, dtype=np.int32)
        counter = 0
        while True:
            if counter % self.user_num == 0:
                np.random.shuffle(uids)
            one_batch = []
            for i in range(self.batch_size):
                one_batch.append(sample(uids[counter % self.user_num]))
                counter += 1
            result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        sampler = ML1MSampler(User, usernum, itemnum, batch_size, maxlen)
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sampler.sample_function,
                    args=(self.result_queue, ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    root_path = os.path.abspath(__file__)
    with open(os.path.join(os.path.sep.join((root_path.split(os.path.sep)[:-1])), fname, f"{fname}.txt")) as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
