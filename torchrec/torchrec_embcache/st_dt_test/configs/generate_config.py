import json
import os
import random
import sys

import yaml

random.seed(4)


def generate_feature_names_list(features, tables):
    feature_names_list = []
    for i in range(tables):
        feature_names_list.append([])
        for j in range(features):
            feature_name = f"features{i*features+j}"
            feature_names_list[-1].append(feature_name)
    return feature_names_list


def generate_num_embeddings(num_embeddings_message):
    num_embedding_list = []
    for message in num_embeddings_message:
        if isinstance(message, list):
            if message[0] == "range":
                num_embedding_list.extend(list(range(*message[1:])))
            elif message[0] == "randint":
                num_embedding_list.append(random.randint(*message[1:]))
        else:
            num_embedding_list.append(message)
    return num_embedding_list


folder_path = sys.argv[1]
with open(os.path.join(folder_path, "test_cases.jsonl")) as f:
    for json_line in json.load(f):
        row = json.loads(json_line)
        test_case_name = row["test_case_name"]
        world_size = row["world_size"]
        table_num = row["table_num"]
        embedding_dims = row["embedding_dims"]
        num_embeddings = row["num_embeddings"]
        pool_type = row["pool_type"]
        batch = row["batch"]
        sharding_type = row["sharding_type"]
        init_fn = row["init_fn"]
        optimizer = row["optimizer"]
        feature_num = row["feature_num"]
        lookup_lens = row["lookup_lens"]
        RecDataset = row["RecDataset"]
        is_bad_case = row["is_bad_case"]

        config = {
            "WORLD_SIZE": world_size,
            "table_num": table_num,
            "embedding_dims": [embedding_dims[0]] * embedding_dims[1], # undo 需要根据参数range部分固定，部分随机设定值，不用专门传参
            "num_embeddings": generate_num_embeddings(num_embeddings),
            "pool_type": pool_type,
            "BATCH_NUM": batch,
            "sharding_type": sharding_type,
            "init_fn": init_fn,
            "optim": optimizer,
            "feature_names_list": generate_feature_names_list(feature_num, table_num),
            "lookup_lens": lookup_lens,
            "RecDataset": RecDataset,
            "is_bad_case": is_bad_case
        }
        if "collection_type" in row:
            config["collection_type"] = row["collection_type"]

        with open(os.path.join(folder_path, f"{test_case_name}.yaml"), 'w') as file:
            yaml.dump(config, file, default_flow_style=False)


