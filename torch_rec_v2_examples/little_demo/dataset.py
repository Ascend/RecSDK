#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from typing import List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from logger import logger


class MovieLensDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train"):
        ratings_file = os.path.join(data_path, "ratings.dat")
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

        ratings_data = []
        with open(ratings_file, "r", encoding="ISO-8859-1") as f:
            for line in f:
                user_id, movie_id, rating, timestamp = line.strip().split("::")
                ratings_data.append(
                    {
                        "user_id": int(user_id),
                        "movie_id": int(movie_id),
                        "rating": float(rating),
                        "timestamp": int(timestamp),
                    }
                )

        ratings_df = pd.DataFrame(ratings_data)

        users_file = os.path.join(data_path, "users.dat")
        movies_file = os.path.join(data_path, "movies.dat")

        users_data = _gen_users_data(users_file)
        users_df = pd.DataFrame(users_data)

        movies_data = _gen_movies_data(movies_file)
        movies_df = pd.DataFrame(movies_data)

        data = pd.merge(ratings_df, users_df, on="user_id", how="left")
        data = pd.merge(data, movies_df, on="movie_id", how="left")

        data = data.sort_values("timestamp")

        split_idx = int(len(data) * 0.8)  # split rate
        if split == "train":
            self.data = data.iloc[:split_idx]
        else:
            self.data = data.iloc[split_idx:]

        self.max_user_id = data["user_id"].max()
        self.max_movie_id = data["movie_id"].max()

        logger.info(f"The length of data: {len(self.data)}.")
        logger.info(
            f"The max user id: {self.max_user_id}, the max movie id: {self.max_movie_id}."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        sparse_features = {
            "user_id": torch.tensor([row["user_id"]], dtype=torch.long),
            "movie_id": torch.tensor([row["movie_id"]], dtype=torch.long),
            "gender": torch.tensor([row["gender"]], dtype=torch.long),
            "age": torch.tensor([row["age"]], dtype=torch.long),
            "occupation": torch.tensor([row["occupation"]], dtype=torch.long),
            "year": torch.tensor([row["year"]], dtype=torch.long),
        }

        label = torch.tensor(row["rating"], dtype=torch.float)

        return sparse_features, label


def collate_fn(batch):
    sparse_features = {
        "user_id": [],
        "movie_id": [],
        "gender": [],
        "age": [],
        "occupation": [],
        "year": [],
    }

    labels = []

    for features, label in batch:
        for key in sparse_features:
            sparse_features[key].extend(features[key].tolist())
        labels.append(label)

    lengths = {
        "user_id": torch.tensor([1] * len(batch), dtype=torch.long),
        "movie_id": torch.tensor([1] * len(batch), dtype=torch.long),
        "gender": torch.tensor([1] * len(batch), dtype=torch.long),
        "age": torch.tensor([1] * len(batch), dtype=torch.long),
        "occupation": torch.tensor([1] * len(batch), dtype=torch.long),
        "year": torch.tensor([1] * len(batch), dtype=torch.long),
    }

    values = {
        "user_id": torch.tensor(sparse_features["user_id"], dtype=torch.long),
        "movie_id": torch.tensor(sparse_features["movie_id"], dtype=torch.long),
        "gender": torch.tensor(sparse_features["gender"], dtype=torch.long),
        "age": torch.tensor(sparse_features["age"], dtype=torch.long),
        "occupation": torch.tensor(sparse_features["occupation"], dtype=torch.long),
        "year": torch.tensor(sparse_features["year"], dtype=torch.long),
    }

    kjt = KeyedJaggedTensor(
        keys=list(values.keys()),
        values=torch.cat(list(values.values())),
        lengths=torch.cat(list(lengths.values())),
    )

    return kjt, torch.tensor(labels, dtype=torch.float)


def _gen_users_data(users_file: str) -> List[Dict[str, int]]:
    users_data = []
    with open(users_file, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            user_id = int(parts[0])
            gender = 1 if parts[1] == "M" else 0
            age = int(parts[2])
            occupation = int(parts[3])
            users_data.append(
                {
                    "user_id": user_id,
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                }
            )
    return users_data


def _gen_movies_data(movies_file: str) -> List[Dict[str, int]]:
    movies_data = []
    with open(movies_file, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            year = 0
            if parts[1].endswith(")"):
                year_start = parts[1].rfind("(")
                if year_start != -1:
                    year_str = parts[1][year_start + 1: parts[1].rfind(")")]
                    try:
                        year = int(year_str)
                    except ValueError:
                        year = 0
            movies_data.append({"movie_id": movie_id, "year": year})
    return movies_data
