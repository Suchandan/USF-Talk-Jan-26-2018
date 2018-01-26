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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

# import input_data_textline as input_data
import input_data_tfrecords as input_data
import tensorflow as tf

import kdd99

# ==============================================================================
# Constants
# ==============================================================================

FLAGS = None
NUM_FEATURES = input_data.num_features
NUM_CLASSES = input_data.num_classes


# ==============================================================================
# Train
# ==============================================================================


def run_training():
    """Train the model."""

    with tf.Graph().as_default():

        step = 0

        # Define the data iterators.
        train_data_iterator = input_data.input_fn(dataset='train', batch_size=FLAGS.batch_size)
        validation_data_iterator = input_data.input_fn(dataset='validation', batch_size=100 * FLAGS.batch_size)
        test_data_iterator = input_data.input_fn(dataset='test', batch_size=100 * FLAGS.batch_size)

        # Define placeholders for data.
        handle, feedable_iterator = input_data.get_feedable_iterator(batch_size=FLAGS.batch_size)
        features_placeholder, labels_placeholder = feedable_iterator.get_next()

        logits = kdd99.inference(input_features=features_placeholder)

        loss = kdd99.loss(logits=logits, labels=labels_placeholder)

        # This Op will calculate and apply gradient descent.
        train_op = kdd99.training(loss=loss, learning_rate=FLAGS.learning_rate)

        # This Op will be used to evaluate the quality of the model.
        eval_correct = kdd99.evaluation(logits=logits, labels=labels_placeholder)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 200
        config.inter_op_parallelism_threads = 200

        sess = tf.Session(config=config)

        # Run the Op to initialize the variables.
        sess.run(init)

        training_time_start = time.time()
        log_num_steps = 1000

        for epoch in range(FLAGS.num_epochs):

            print("Training epoch: ", epoch)
            sess.run(train_data_iterator.initializer)
            sess.run(validation_data_iterator.initializer)
            sess.run(test_data_iterator.initializer)

            training_handle = sess.run(train_data_iterator.string_handle())
            validation_handle = sess.run(validation_data_iterator.string_handle())

            epoch_time_start = time.time()
            step_collection_time_start = time.time()

            epoch_num_correct = 0
            epoch_num_samples = 0

            #########################################################################################################
            # Training
            #########################################################################################################

            while True:
                try:

                    _, num_correct, loss_value, features_batch, labels_batch = sess.run(
                        [train_op, eval_correct, loss, features_placeholder, labels_placeholder],
                        feed_dict={handle: training_handle})
                    step += 1

                    epoch_num_correct += num_correct
                    epoch_num_samples += len(features_batch)
                    epoch_accuracy = 100 * float(epoch_num_correct) / float(epoch_num_samples)

                    if step % log_num_steps == 0:
                        duration = (time.time() - step_collection_time_start) / 60.0

                        # Reset step_collection_time_start
                        step_collection_time_start = time.time()

                        # Print status to stdout.
                        print(
                            'Step %d: loss = %.2f; (time for %d steps : %.3f minutes). Epoch accuracy so far: %0.5f' % (
                                step, loss_value, log_num_steps, duration, epoch_accuracy))

                except tf.errors.OutOfRangeError:
                    break

            print(
                "Completed training epoch %d in %.2f minutes. Total time elapsed: %.2f minutes. Epoch accuracy: %.2f %%" % (
                    epoch,
                    (time.time() - epoch_time_start) / 60.0,
                    (time.time() - training_time_start) / 60.0,
                    epoch_accuracy))

            #########################################################################################################
            # Validation
            #########################################################################################################

            validation_time_start = time.time()
            num_validation_samples = 0
            num_validation_correct = 0
            while True:
                try:
                    num_correct, features_batch = sess.run([eval_correct, features_placeholder],
                                                           feed_dict={handle: validation_handle})
                    num_validation_correct += num_correct
                    num_validation_samples += len(features_batch)

                except tf.errors.OutOfRangeError:
                    break

            # Print status to stdout.
            validation_time = (time.time() - validation_time_start) / 60.0
            validation_accuracy = 100 * float(num_validation_correct) / num_validation_samples

            print('Validation num_correct: %d Num samples: %d Time taken (minutes): %.02f' % (
                num_validation_correct, num_validation_samples, validation_time))
            print("Validation Accuracy: %.02f %%" % validation_accuracy)

        #########################################################################################################
        # Test
        #########################################################################################################

        test_handle = sess.run(test_data_iterator.string_handle())

        test_time_start = time.time()
        num_test_samples = 0
        num_test_correct = 0
        while True:
            try:
                num_correct, features_batch = sess.run([eval_correct, features_placeholder],
                                                       feed_dict={handle: test_handle})
                num_test_correct += num_correct
                num_test_samples += len(features_batch)

            except tf.errors.OutOfRangeError:
                break

        # Print status to stdout.
        test_time = (time.time() - test_time_start) / 60.0
        test_accuracy = 100 * float(num_test_correct) / num_test_samples

        print('Test num_correct: %d Num samples: %d Time taken (minutes): %.02f' % (
            num_test_correct, num_test_samples, test_time))
        print("Test Accuracy: %.02f%%" % test_accuracy)

        sess.close()


# ==============================================================================
# Logging
# ==============================================================================

def reset_log_dir():
    if tf.gfile.Exists(FLAGS.log_dir):
        print("Deleting old log directory.")
        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    print("Making log directory.")
    tf.gfile.MakeDirs(FLAGS.log_dir)


# ==============================================================================
# Main
# ==============================================================================

def main():
    reset_log_dir()
    run_training()


# ==============================================================================
# __name__ == __main__
# ==============================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='Number of epochs to run trainer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
