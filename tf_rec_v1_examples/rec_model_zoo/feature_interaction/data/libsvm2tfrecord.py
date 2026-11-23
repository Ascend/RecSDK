import os
import stat
import logging
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_tfrecords(input_filename, output_filename):
    """Concert the LibSVM contents to TFRecord.
    Args:
    input_filename: LibSVM filename.
    output_filename: Desired TFRecord filename.
    """
    logging.info("Starting to convert {} to {}...".format(input_filename, output_filename))
    writer = tf.io.TFRecordWriter(output_filename)
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(input_filename, flags, modes), "r") as file:
        for line in file:
            data = line.split(" ")
            label = float(data[0])
            ids = []
            values = []
            for fea in data[1:]:
                id_, value_ = fea.split(":")
                ids.append(int(id_))
                values.append(float(value_))
            # Write samples one by one
            example = tf.train.Example(features=tf.train.Features(feature={
                "label":
                    tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                "ids":
                    tf.train.Feature(int64_list=tf.train.Int64List(value=ids)),
                "values":
                    tf.train.Feature(float_list=tf.train.FloatList(value=values))
            }))
            writer.write(example.SerializeToString())
        writer.close()
        logging.info("Successfully converted {} to {}!".format(input_filename, output_filename))


sess = tf.compat.v1.InteractiveSession()
convert_tfrecords("./criteo/train.libsvm", "./criteo/train.tfrecords")
convert_tfrecords("./criteo/valid.libsvm", "./criteo/valid.tfrecords")
convert_tfrecords("./criteo/test.libsvm", "./criteo/test.tfrecords")
sess.close()
