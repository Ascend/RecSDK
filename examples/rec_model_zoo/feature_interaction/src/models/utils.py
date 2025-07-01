import re
import os
import stat
import glob
from typing import Dict, List, Tuple
import tensorflow as tf


# ------ Load tfrecord dataset ------
def input_fn(filenames: List[str], batch_size: int = 32, field_size: int = 39, num_epochs: int = 1,
             perform_shuffle: bool = False) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """
    Input function for loading TFRecord dataset.

    Args:
        filenames (List[str]): List of TFRecord file paths.
        batch_size (int): Batch size.
        field_size (int): Number of fields.
        num_epochs (int): Number of epochs to repeat the dataset.
        perform_shuffle (bool): Whether to shuffle the dataset.

    Returns:
        Tuple[Dict[str, tf.Tensor], tf.Tensor]: Batch features and batch labels.
    """

    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
            'ids': tf.io.FixedLenFeature(shape=(field_size,), dtype=tf.int64),
            'values': tf.io.FixedLenFeature(shape=(field_size,), dtype=tf.float32),
        }
        sample = tf.io.parse_example(data_record, features)
        sample['ids'] = tf.cast(sample['ids'], dtype=tf.int32)
        return {"feat_ids": sample['ids'], "feat_vals": sample['values']}, sample['label']

    dataset = tf.data.TFRecordDataset(filenames)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True).map(extract_fn, num_parallel_calls=10).prefetch(100)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def build_optimizer(optimizer_name: str, learning_rate: float) -> tf.compat.v1.train.Optimizer:
    """
    Build the optimizer.

    Args:
        optimizer_name (str): Name of the optimizer.
        learning_rate (float): Learning rate.

    Returns:
        tf.compat.v1.train.Optimizer: Optimizer.
    """
    if optimizer_name == 'Adam':
        return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer_name == 'Adagrad':
        return tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer_name == 'Momentum':
        return tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif optimizer_name == 'ftrl':
        return tf.compat.v1.train.FtrlOptimizer(learning_rate)
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_name))


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
    flags = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    pred_path = os.path.join(data_dir, "pred.txt")
    with os.fdopen(os.open(pred_path, flags, modes), "w") as fo:
        for prob in preds:
            fo.write("%f\n" % (prob['prob']))
