# ==============================================================================
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
"""Implements an input pipeline with tf.data.TextLineDataset
"""

from __future__ import print_function

import tensorflow as tf

from helper_functions import *


# ==============================================================================
# Make tf.data.dataset
# ==============================================================================

def get_feature_dict_and_label_from_line(line, feature_names=numeric_features):
    record_defaults = [[0.0] for _ in feature_names] + [[0]]
    parsed_line = tf.decode_csv(line, record_defaults=record_defaults)

    # The label is the last element
    label = tf.cast(parsed_line[-1], tf.int32)
    label = tf.one_hot(label, depth=num_classes)

    del parsed_line[-1]
    features = parsed_line

    return features, label


def get_dataset(dataset, batch_size, perform_shuffle=False):
    file_fullpath = get_fullpath_for_dataset(dataset=dataset)
    dataset = (tf.data.TextLineDataset(file_fullpath, compression_type='GZIP').
        cache().
        skip(1).
        map(get_feature_dict_and_label_from_line))

    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.batch(batch_size)
    return dataset


# ==============================================================================
# Make tensorflow iterators
# ==============================================================================

def input_fn(dataset, batch_size=100, perform_shuffle=False):
    dataset = get_dataset(dataset, perform_shuffle=perform_shuffle, batch_size=batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator


def get_feedable_iterator(batch_size):
    _ = get_dataset("train", batch_size=batch_size, perform_shuffle=False)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, _.output_types, _.output_shapes)

    return handle, iterator
