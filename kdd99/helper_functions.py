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

from __future__ import print_function

import os

import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# Constants
# ==============================================================================

raw_data_dir = "./data/raw_data"
input_data_dir = "./data/split_data"
tfrecords_input_data_dir = "./data/split_data/tfrecords"

# ==============================================================================
# Header
# ==============================================================================


features = ['duration',
            'protocol_type',
            'service',
            'flag',
            'src_bytes',
            'dst_bytes',
            'land',
            'wrong_fragment',
            'urgent',
            'hot',
            'num_failed_logins',
            'logged_in',
            'num_compromised',
            'root_shell',
            'su_attempted',
            'num_root',
            'num_file_creations',
            'num_shells',
            'num_access_files',
            'num_outbound_cmds',
            'is_host_login',
            'is_guest_login',
            'count',
            'srv_count',
            'serror_rate',
            'srv_serror_rate',
            'rerror_rate',
            'srv_rerror_rate',
            'same_srv_rate',
            'diff_srv_rate',
            'srv_diff_host_rate',
            'dst_host_count',
            'dst_host_srv_count',
            'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate',
            'dst_host_srv_serror_rate',
            'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate']

label_column_name = "label"
string_features = ['land', 'service', 'protocol_type', 'flag']
sumbolic_features = ["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login"]

numeric_features = [col for col in features if col not in sumbolic_features]
column_names = features + [label_column_name]
num_features = len(numeric_features)
num_classes = 2

features_key = "features"


# ==============================================================================
# Get path of files
# ==============================================================================

def get_fullpath_for_dataset(dataset, check_exists=False):
    assert dataset in ['train', 'validation', 'test'], "Bad input: dataset"

    file_fullpath = os.path.join(input_data_dir, dataset + ".csv.gz")

    if check_exists:
        assert os.path.exists(file_fullpath), "File doesn't exist!"
    return file_fullpath


# ==============================================================================
# Get path of files
# ==============================================================================

def get_fullpath_for_tfrecords_dataset(dataset, check_exists=False):
    file_fullpath = os.path.join(tfrecords_input_data_dir, dataset + ".tfrecords.gz")

    if check_exists:
        assert os.path.exists(file_fullpath), "File doesn't exist!"

    if file_fullpath[-3:] == ".gz":
        compression_type = "GZIP"
    else:
        compression_type = None

    return compression_type, file_fullpath


# ==============================================================================
# Make Test Train Split
# ==============================================================================

def kdd99_test_train_split(train_size=.6, nrows=None):
    print("Number of rows of KDD99 data: %s (if None, read all)" % str(nrows))
    print("Starting to make train/validation/test split ...", )

    file_fullpath = os.path.join(raw_data_dir, "kddcup.data.gz")

    usecols = numeric_features + [label_column_name]
    df = pd.read_csv(file_fullpath, header=None, names=column_names, usecols=usecols, nrows=nrows)

    df[label_column_name] = (df[label_column_name].values == 'normal.')
    df[label_column_name] = df[label_column_name].astype(int)

    df = df[numeric_features + [label_column_name]]

    print("Standardizing data ...", )
    for feature in numeric_features:
        std = df[feature].std()
        mean = df[feature].mean()
        df.loc[:, feature] = (df[feature] - mean) / std
    print("done.")

    X_train, X_test = train_test_split(df, test_size=1 - train_size)
    X_validation, X_test = train_test_split(X_test, test_size=0.5)

    # Throw away data to account for batch size:
    num_train_rows = 100 * int(len(X_train) / 100)
    num_validation_rows = 100 * int(len(X_validation) / 100)
    num_test_rows = 100 * int(len(X_test) / 100)

    train_fullpath = get_fullpath_for_dataset(dataset='train', check_exists=False)
    validation_fullpath = get_fullpath_for_dataset(dataset='validation', check_exists=False)
    test_fullpath = get_fullpath_for_dataset(dataset='test', check_exists=False)

    X_train[:num_train_rows].to_csv(train_fullpath, index=False, compression='gzip')
    X_validation[:num_validation_rows].to_csv(validation_fullpath, index=False, compression='gzip')
    X_test[:num_test_rows].to_csv(test_fullpath, index=False, compression='gzip')

    print("done.")
