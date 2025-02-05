import json
import os
import stat
import argparse
import math

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length
flags = os.O_RDONLY
modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
with os.fdopen(os.open("keymap_train.json", flags, modes), "r") as fp:
    obj = json.load(fp)
new_dict = dict()
for key in obj.keys():
    arr = []
    for k, v in obj[key].items():
        if v <= 1:
            continue
        arr.append(int(k))
    arr.sort()
    new_dict[key] = arr

json_str = json.dumps(new_dict, indent=4, sort_keys=True)
flags = os.O_WRONLY | os.O_CREAT
modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
with os.fdopen(os.open("keymap_train_pruned.json", flags, modes), "w") as json_file:
    json_file.write(json_str)

for key in obj.keys():
    if not os.path.exists("./aliccp_out/vocab"):
        os.makedirs("./aliccp_out/vocab")
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(os.path.join("./aliccp_out/vocab/vocab_", key), flags, modes), "w") as fp:
        fp.writelines([f"{val}\n" for val in new_dict[key]])
