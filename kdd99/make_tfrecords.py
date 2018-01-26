from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import time

import pandas as pd
import ray
import tensorflow as tf
from helper_functions import *

# ==============================================================================
# Constants
# ==============================================================================

tfrecords_output_dir_name = "tfrecords"


# ==============================================================================
# Call save_tfrecords via Ray
# ==============================================================================

def convert_directory():
    time_start = time.time()

    files = glob.glob(os.path.join(FLAGS.directory, FLAGS.pattern))
    output_dir = os.path.join(FLAGS.directory, tfrecords_output_dir_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    object_ids = []
    for file in files:
        _, tail = os.path.split(file)
        filename = tail.split(".")[0]
        filename = filename + ".tfrecords.gz"

        output_file_fullpath = os.path.join(output_dir, filename)

        object_id = save_tfrecords.remote(file, output_file_fullpath)
        object_ids.append(object_id)

    results = ray.get(object_ids)

    duration = (time.time() - time_start) / 60.0
    print("Done processing %s in %.2f minutes." % (FLAGS.directory, duration))


# ==============================================================================
# Helper functions for TFRecords
# ==============================================================================

def _float32_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_scalar_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


# ==============================================================================
# Convert Pandas DataFrame to TFRecords and save.
# ==============================================================================

@ray.remote
def save_tfrecords(input_file_fullpath, output_file_fullpath):
    """
    :param input_file_fullpath: Input dataframe. Assumed to be a labeled dataset, the last column is the label (int64)
    :param output_file_fullpath: Where to save the output file. If the extension is .gz, then it is compressed.
    :return: None
    """

    print("Reading: {}".format(input_file_fullpath))
    df = pd.read_csv(input_file_fullpath, usecols=numeric_features + [label_column_name])
    print("Done reading : {}".format(input_file_fullpath))

    # Warning: Must handle NA values, since default values of tf.FixedLenFeature()
    # are not applied if an individual entry of a row is missing.

    df = df.fillna(0.0)
    print("Filled NA Values with 0: {}".format(input_file_fullpath))

    if output_file_fullpath[-3:] == ".gz":
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    else:
        options = None

    time_start = time.time()

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        with tf.python_io.TFRecordWriter(output_file_fullpath, options=options) as writer:

            for i, (_, _row) in enumerate(df.iterrows()):

                row = _row.values
                if i % 1000 == 0:
                    print(output_file_fullpath, i)
                features, label = row[:-1], int(row[-1])

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        features_key: _float32_array_feature(features),
                        label_column_name: _int64_scalar_feature(label)
                    }))

                writer.write(example.SerializeToString())

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)

    duration = (time.time() - time_start) / 60.0
    print("Total time taken to convert and write %s : %.2f minutes" % (output_file_fullpath, duration))


# ==============================================================================
# Main
# ==============================================================================

def main():
    ray.init()
    convert_directory()


# ==============================================================================
# __name__ == "__main__"
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pattern',
        type=str,
        default="*.gz",
        help='Glob pattern of files to read'
    )

    parser.add_argument(
        '--directory',
        type=str,
        default="./",
        help='Directory to grab input files from'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
