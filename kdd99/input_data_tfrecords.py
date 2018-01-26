# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements an input pipeline with tf.data.TFRecordDataset
"""

from __future__ import print_function

import tensorflow as tf

from helper_functions import *


# ==============================================================================
# Make tf.data.dataset from TFRecords
# ==============================================================================

def get_dataset_tfrecords(dataset, batch_size, perform_shuffle=False, one_hot=True):
    compression_type, file_fullpath = get_fullpath_for_tfrecords_dataset(dataset=dataset)

    keys_to_features = {
        label_column_name: tf.FixedLenFeature(shape=(),
                                              dtype=tf.int64,
                                              default_value=tf.zeros(shape=(), dtype=tf.int64)
                                              ),

        features_key: tf.FixedLenFeature(shape=(len(numeric_features)),
                                         dtype=tf.float32,
                                         default_value=tf.zeros(shape=(len(numeric_features)), dtype=tf.float32)
                                         )
    }

    def parser(record):

        parsed = tf.parse_single_example(record, keys_to_features)
        label = parsed[label_column_name]

        if one_hot:
            label = tf.one_hot(label, depth=num_classes)

        return parsed[features_key], label

    dataset = (tf.data.TFRecordDataset(file_fullpath, compression_type=compression_type, buffer_size=200 * 1024 * 1024)
        .map(parser, num_parallel_calls=10)
        .prefetch(500000)
        )

    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.batch(batch_size)
    return dataset


# ==============================================================================
# Make tensorflow iterators
# ==============================================================================

def input_fn(dataset, batch_size=100, perform_shuffle=False):
    dataset = get_dataset_tfrecords(dataset, perform_shuffle=perform_shuffle, batch_size=batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator


def get_feedable_iterator(batch_size):
    _ = get_dataset_tfrecords("train", batch_size=batch_size, perform_shuffle=False)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, _.output_types, _.output_shapes)

    return handle, iterator
