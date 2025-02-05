import re
import os
import stat
import glob
from typing import Dict, List


def get_third_nearest_checkpoint(path):
    filenames = glob.glob(os.path.join(path, 'model.ckpt-*.index'))
    pattern = re.compile(r'model.ckpt-(.*?).index', re.S)
    versions = []
    for file in filenames:
        versions += [int(re.findall(pattern, file)[0])]
    versions_sorted = sorted(versions)
    return os.path.join(path, 'model.ckpt-' + str(versions_sorted[-3]))


def dump_pred(preds: List[Dict[str, float]], data_dir: str) -> None:
    """
    Dump the prediction results to a file.

    Args:
        preds (List[Dict[str, float]]): List of prediction results.
        data_dir (str): data save dir.

    Returns:
        None
    """
    flags = os.O_WRONLY | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(data_dir, "pred.txt")
    with os.fdopen(os.open(pred_path, flags, modes), "w") as fo:
        for prob in preds:
            fo.write("%f\n" % (prob['prob']))
