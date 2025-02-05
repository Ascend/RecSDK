import argparse
import math
import json
import os
import stat
from multiprocessing import Process

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length

fields = [
    "101", "109_14", "110_14", "127_14", "150_14", "121", "122", "124", "125", "126", "127", "128", "129",
    "205", "206", "207", "210", "216", "508", "509", "702", "853", "301",
]


def merge_data(common_file_name: str, skeleton_file_name: str, out_file_name: str):
    fields_ = [
        "101", "109_14", "110_14", "127_14", "150_14", "121", "122", "124", "125", "126", "127", "128", "129",
        "205", "206", "207", "210", "216", "508", "509", "702", "853", "301",
    ]

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(out_file_name, flags, modes), "w") as write_file:
        common_dict: dict[str, str] = dict()
        max_length_dict: dict[str, int] = dict(map(lambda s: [s, 0], fields_))

        flags = os.O_RDONLY
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        with os.fdopen(os.open(common_file_name, flags, modes), "r") as f:
            line_count = 0
            while True:
                lines = f.readlines(10000000000)
                if len(lines) == 0:
                    break
                line_count += len(lines)
                for line in lines:
                    cells = line.strip().split(",")
                    common_dict[cells[0]] = cells[1:]

        with os.fdopen(os.open(skeleton_file_name, flags, modes), "r") as f:
            line_count = 0
            while True:
                lines = f.readlines(1000000000)
                if len(lines) == 0:
                    break
                line_count += len(lines)
                lines_to_write = []
                for line in lines:
                    cells = line.strip().split(",")
                    all_feats = cells[4:] + common_dict[cells[3]]
                    local_dict = dict()
                    for feat in all_feats:
                        key, value = feat.split(":")
                        local_dict[key] = value
                    for field in fields_:
                        if field not in local_dict:
                            local_dict[field] = "0"
                        max_length_dict[field] = max(max_length_dict[field], len(local_dict[field].split("#")))
                    strs = []
                    for field in fields_:
                        strs.append(local_dict[field])
                    lines_to_write.append(",".join(cells[:3] + strs) + "\n")
                write_file.writelines(lines_to_write)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(out_file_name.replace(".csv", "_max_length.json"), flags, modes), "w") as fp:
        json.dump(max_length_dict, fp, indent=4)


if __name__ == "__main__":
    tasks = [
        ("./common_features_train_parsed.csv", "./sample_skeleton_train_parsed.csv", "./data_train.csv"),
        ("./common_features_test_parsed.csv", "./sample_skeleton_test_splitted_parsed.csv", "./data_test.csv"),
        ("./common_features_test_parsed.csv", "./sample_skeleton_val_splitted_parsed.csv", "./data_val.csv"),
    ]
    ps = [Process(target=merge_data, args=task) for task in tasks]
    for p in ps:
        p.start()
    for p in ps:
        p.join()