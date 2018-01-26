from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

logger = logging.getLogger(__name__)

#######################################################
# Constants
#######################################################

ACTIVITY_LABELS_FULLPATH = './data/activity_labels.txt'

RAW_DATA_DIR = "./data/{}/Inertial Signals/"

FEATURE_FILES = [
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt"
]

LABELS_FULLPATH = "./data/{}/y_{}.txt"

VALIDATION_FRACTION = .1

NUM_CLASSES = 6
NUM_FEATURES = 9


#######################################################
# Sanity Checks
#######################################################

class Data:

    #######################################################
    # Init
    #######################################################

    def __init__(self):

        self.read_all_data()
        self.get_test_train_validation_split()

    #######################################################
    # Filenames
    #######################################################

    def get_feature_fullpaths(self, dataset):

        result = [os.path.join(RAW_DATA_DIR.format(dataset), _.format(dataset)) for _ in FEATURE_FILES]
        return result

    def get_labels_fullpath(self, dataset):

        result = LABELS_FULLPATH.format(dataset, dataset)

        return result

    #######################################################
    # Read Data
    #######################################################

    def get_activity_labels(self):
        """
        Read activity labels
        :return:
        """
        activity_labels = u""

        with open(ACTIVITY_LABELS_FULLPATH) as f:
            for line in f:
                activity_labels += line.strip() + "\n"

        activity_labels = io.StringIO(activity_labels)
        activity_labels = pd.read_csv(activity_labels, sep=" ", header=None, names=["id", "name"])
        activity_labels.loc[:, ['id']] = activity_labels['id'] - 1

        self.activity_labels = activity_labels
        return self.activity_labels

    def get_labels(self, dataset):
        """
        Read Labels for RawData. Modify activity labels to be integers in [0, NUM_CLASSES)
        """
        assert dataset in ['train', 'test'], "Bad Input!"

        file_fullpath = self.get_labels_fullpath(dataset=dataset)

        labels = pd.read_csv(file_fullpath, sep=r"\s*", header=None, names=['label'], engine='python')
        labels.loc[:, 'label'] = labels['label'] - 1

        # Fit one hot encoder
        if dataset == 'train':
            self.ohe = OneHotEncoder(n_values=NUM_CLASSES)
            n = len(labels.label.values)
            labels = labels.label.values.reshape(n, 1)
            labels = self.ohe.fit_transform(labels).todense()

        if dataset == 'test':

            if not hasattr(self, 'ohe'):
                print("Get Labels for train before test!")
                raise RuntimeError

            n = len(labels.label.values)
            labels = labels.label.values.reshape(n, 1)
            labels = self.ohe.transform(labels).todense()

        setattr(self, 'y_{}'.format(dataset), labels)

        return None

    def get_raw_data(self, dataset):
        """
        Read raw data from disk.
        :return:
        """

        features = []

        files = self.get_feature_fullpaths(dataset)

        logger.info("Reading raw data ...")
        for file_fullpath in files:
            logger.info("Reading {}".format(file_fullpath))
            matrix = pd.read_csv(file_fullpath, sep=r"\s*", header=None, engine='python')
            matrix = matrix.as_matrix()
            features.append(matrix)

        logger.info("Done reading raw data.")
        logger.info("Names of features: \n {}".format("\n\t".join(files)))

        features = np.stack(features, axis=2)

        # Transpose so the shape is (time index, sample number, feature number)
        features = features.transpose((1, 0, 2))

        setattr(self, "X_{}".format(dataset), features)

        return None

    def read_all_data(self):

        self.get_activity_labels()

        self.get_raw_data("train")
        self.get_raw_data("test")

        self.get_labels('train')
        self.get_labels('test')

    #######################################################
    # Make Data
    #######################################################

    # features_placeholder = tf.placeholder(tf.float32, (None, None, NUM_FEATURES))  # (time, batch, num_features)
    # labels_placeholder = tf.placeholder(tf.float32, (None, NUM_CLASSES))  # (batch, out)

    #######################################################
    # Make Test/Train/Validation Split
    #######################################################

    def get_test_train_validation_split(self, VALIDATION_FRACTION=VALIDATION_FRACTION):
        """
        :param keys: A list of tuples of the form [(..., label)]
        :param VALIDATION_FRACTION: Fraction of data in validation set
        :return: A tuple of disjoing lists train_keys, validation_keys whose union is keys. Here, validation_keys contains approximately
        VALIDATION_FRACTION of each of the the keys for each possible label in keys.
        """

        grouped_indices = defaultdict(list)
        validation_indices = []

        y_train = getattr(self, "y_train")
        for index in range(y_train.shape[0]):
            row = y_train[index]
            label = np.argmax(row)

            grouped_indices[label].append(index)
            assert label in range(0, NUM_CLASSES), "Error! Bad data or label read incorrectly."

        for label, indices in grouped_indices.items():
            random.shuffle(indices)

            validation_indices_current_label = indices[: int(VALIDATION_FRACTION * len(indices))]
            validation_indices.extend(validation_indices_current_label)

            logger.info("Number of training samples with label {}: {}".format(label,
                                                                              len(indices) - len(
                                                                                  validation_indices_current_label)
                                                                              ))
            logger.info(
                "Number of validation samples with label {}: {}".format(label, len(validation_indices_current_label)))

        train_indices = [_ for _ in range(y_train.shape[0]) if _ not in validation_indices]

        # Update X_train, y_train, X_validation, y_validation
        self.X_validation = self.X_train[:, validation_indices, :]
        self.y_validation = self.y_train[validation_indices, :]

        self.X_train = self.X_train[:, train_indices, :]
        self.y_train = self.y_train[train_indices, :]

        logger.info("Training size: {}\nValidation size: {} ".format(len(train_indices), len(validation_indices)))

        return None

    #######################################################
    # Generate Batch Data
    #######################################################

    def batch_data_iterator(self, dataset, batch_size, do_shuffle=False):
        """
        :param dataset: Name of the dataset
        :param batch_size: Batch size
        :param do_shuffle: Shuffle data before iterating.
        :return: Batches of data, of size batch_size as pairs of numpy arrays.
        """

        _batch_times = []

        X, y = self.get_dataset(dataset, do_shuffle=do_shuffle)
        n = len(y)

        X_batches = np.array_split(X, n / batch_size, axis=1)
        y_batches = np.array_split(y, n / batch_size, axis=0)

        batches = zip(X_batches, y_batches)

        for batch in batches:
            time_start = time.time()
            yield batch

            _batch_times.append(time.time() - time_start)
        print("Average time to process batch of size {} is {} seconds".format(batch_size, np.mean(_batch_times)))

    def get_dataset(self, dataset, do_shuffle=False):

        X = getattr(self, "X_{}".format(dataset))
        y = getattr(self, "y_{}".format(dataset))

        if do_shuffle == True:
            indices = range(len(y))
            np.random.shuffle(indices)

            X = X[:, indices, :]
            y = y[indices, :]

        return X, y

    #######################################################
    # Iterators
    #######################################################

    def train_data_iterator(self, batch_size, do_shuffle=False):
        return self.batch_data_iterator(dataset="train", batch_size=batch_size, do_shuffle=do_shuffle)

    def validation_data_iterator(self, batch_size, do_shuffle=False):
        return self.batch_data_iterator(dataset="validation", batch_size=batch_size, do_shuffle=do_shuffle)

    def test_data_iterator(self, batch_size, do_shuffle=False):
        return self.batch_data_iterator(dataset="test", batch_size=batch_size, do_shuffle=do_shuffle)
