import os
import stat
import argparse
import math
import json
from multiprocessing import Process
from collections import defaultdict
from examples.util.path_validator import validate_save_path, validate_read_file

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length


def parse_data(file_name, write_name):
    validate_read_file("./keymap_train_pruned.json")
    with open("./keymap_train_pruned.json") as f:
        map_dict: dict[str, list[int]] = json.load(f)
    map_map_dict: dict[str, dict[str, int]] = dict()
    for field, l_val in map_dict.items():
        map_map_dict[field] = dict([(str(value), index) for index, value in enumerate(l_val)])
    map_dict: dict[str, dict[str, int]] = map_map_dict
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    validate_save_path(write_name)
    with os.fdopen(os.open(write_name, flags, modes), "w") as write_file:
        line_count = 0
        flags = os.O_RDONLY
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        validate_read_file(file_name)
        with os.fdopen(os.open(file_name, flags, modes), "r") as f:
            while True:
                lines = f.readlines(int(1e9))
                if len(lines) == 0:
                    break
                line_count += len(lines)
                lines_to_write = []
                for line in lines:
                    field_dict = defaultdict(int)
                    line = line.strip().split(",")
                    feat_len = len(line)
                    # common_feature_index|feat_num|feat_list
                    if feat_len == 3:
                        sample_dict = process_feat_strs(line[2], map_dict, field_dict)
                        str_buffer = []
                        for k, v in sample_dict.items():
                            value_str = "#".join([str(num) for num in v])
                            str_buffer.append(f"{k}:{value_str}")
                        lines_to_write.append(f"{line[0]},{','.join(str_buffer)}\n")

                    # sample_id|y|z|common_feature_index|feat_num|feat_list
                    elif feat_len == 6:
                        # Skip samples where y is 0 and z is 1
                        if line[1] == "0" and line[2] == "1":
                            continue
                        sample_dict = process_feat_strs(line[5], map_dict, field_dict)
                        str_buffer = []
                        for k, v in sample_dict.items():
                            value_str = "#".join([str(num) for num in v])
                            str_buffer.append(f"{k}:{value_str}")
                        lines_to_write.append(
                            f"{line[0]},{line[1]},{line[2]},{line[3]},{','.join(str_buffer)}\n"
                        )
                write_file.writelines(lines_to_write)
        write_file.close()


def process_feat_strs(feat_strs: str, map_dict: dict[str, dict[str, int]],
                      field_dict: dict) -> dict[str, list[int]]:
    sample_dict = {}
    for fstr in feat_strs.split("\x01"):
        filed, feat_val = fstr.split("\x02")
        feat, val = feat_val.split("\x03")
        if field_dict[filed] >= args.length:
            continue
        field_dict[filed] += 1
        if filed not in sample_dict:
            sample_dict[filed] = []
        feat_mapped = map_dict[filed][feat] + 1 if feat in map_dict[filed].keys() else 0
        sample_dict[filed].append(feat_mapped)
    return sample_dict


if __name__ == "__main__":
    tasks = [
        ("./sample_skeleton_train.csv", "./sample_skeleton_train_parsed.csv"),
        ("./common_features_train.csv", "./common_features_train_parsed.csv"),
        ("./sample_skeleton_test.csv", "./sample_skeleton_test_parsed.csv"),
        ("./common_features_test.csv", "./common_features_test_parsed.csv"),
    ]
    ps = [Process(target=parse_data, args=task) for task in tasks]
    for p in ps:
        p.start()
    for p in ps:
        p.join()