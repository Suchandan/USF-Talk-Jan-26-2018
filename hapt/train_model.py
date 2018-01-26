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

"""Trains and Evaluates the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time

import input_data
import tensorflow as tf

import hapt

# from tensorflow.python import debug as tf_debug

# ==============================================================================
# Constants
# ==============================================================================

logger = logging.getLogger(__name__)
FLAGS = None
NUM_FEATURES = input_data.NUM_FEATURES
NUM_CLASSES = input_data.NUM_CLASSES

# Config for tensorflow session
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 300
config.inter_op_parallelism_threads = 300

NUM_EPOCHS_CHECKPOINT = 5


# ==============================================================================
# Placeholders
# ==============================================================================

def get_placeholders():
    """
    :return: tensorflow placeholders for input and labels
    """

    # Input/Output placeholders
    features_placeholder = tf.placeholder(tf.float32, (None, None, NUM_FEATURES))  # (time, batch, num_features)
    labels_placeholder = tf.placeholder(tf.float32, (None, NUM_CLASSES))  # (time, batch, out)

    return features_placeholder, labels_placeholder


# ==============================================================================
# Main function to do training
# ==============================================================================

def run_training(log_dir,
                 lr_initial,
                 lr_decay_steps,
                 num_epochs,
                 batch_size,
                 model_type,
                 dropout_keep_prob,
                 num_epochs_checkpoint,
                 l2_penalty_multiplier,
                 load_model_fullpath=None
                 ):
    """Train and evaluate the model

    Start tensorboard with the log directory on port 6006

    user@REMOTE_IP: tensorboard --log_dir FLAGS.log_dir

    #The following command will forward port 6006 to REMOTE_IP for testing purposes:

    user@LOCAL_IP: ssh -N -f -L localhost:6006:localhost:6006 REMOTE_IP
    """

    #Initialize variables
    training_loss, validation_loss, train_accuracy, validation_accuracy, test_accuracy = 0, 0, 0, 0, 0
    model_save_path = ""

    time_start = time.time()
    data = input_data.Data()

    logger.info("Saving tensorboard output to log_dir: %s" % log_dir)

    with tf.Graph().as_default() as default_graph:

        features_placeholder, labels_placeholder = get_placeholders()

        logits_function = getattr(hapt, model_type)
        logits = logits_function(input_features=features_placeholder, dropout_keep_prob=dropout_keep_prob)

        loss = hapt.loss(labels=labels_placeholder, logits=logits, l2_penalty_multiplier=l2_penalty_multiplier)
        learning_rate, train_fn, global_step = hapt.training(lr_initial=lr_initial, lr_decay_steps=lr_decay_steps,
                                                             loss=loss)
        correct_prediction = hapt.evaluation(labels=labels_placeholder, logits=logits)

        train_writer = tf.summary.FileWriter(log_dir + '/train',
                                             default_graph)

        validation_writer = tf.summary.FileWriter(log_dir + '/validation',
                                                  default_graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test',
                                                  default_graph)

        summary = tf.summary.merge_all()
        sess = tf.Session(config=config)
        saver = tf.train.Saver()

        if load_model_fullpath == None:
            initializer = tf.global_variables_initializer()
            sess.run(initializer)
        else:
            logger.info("Loading model from : {}".format(load_model_fullpath))
            saver.restore(sess, save_path=load_model_fullpath)
            logger.info("Done loading model.")

        print("Graph built. Starting training. Time taken so far {} minutes".format((time.time() - time_start) / 60.0))

        for epoch in range(1, num_epochs + 1):

            training_loss = 0
            training_num = 0
            training_num_correct = 0
            batch_num = 0

            epoch_time_start = time.time()
            for batch_x, batch_y in data.train_data_iterator(batch_size=batch_size, do_shuffle=True):
                batch_loss, num_correct, train_summary, train_step, _ = sess.run(
                    [loss, correct_prediction, summary, global_step, train_fn], {
                        features_placeholder: batch_x,
                        labels_placeholder: batch_y,
                    })

                train_writer.add_summary(train_summary, train_step)
                batch_num += 1

                training_num += len(batch_y)
                training_loss += batch_loss
                training_num_correct += num_correct

                train_accuracy = training_num_correct / float(training_num)

                train_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_accuracy',
                                                             simple_value=train_accuracy)])
                train_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_average_loss',
                                                             simple_value=training_loss/float(training_num))])

                train_writer.add_summary(train_accuracy_summary, train_step)

                #Only print the loss after the epoch is over, since this isn't average loss - it's total loss.
                train_writer.add_summary(train_loss_summary, train_step)

            ########################################################################
            # Validation
            ########################################################################
            validation_num = 0
            validation_num_correct = 0
            validation_loss = 0
            validation_accuracy = 0

            for batch_x, batch_y in data.validation_data_iterator(batch_size=batch_size, do_shuffle=False):
                num_correct, batch_loss, validation_summary = sess.run([correct_prediction, loss, summary], {
                    features_placeholder: batch_x,
                    labels_placeholder: batch_y,
                })

                validation_writer.add_summary(validation_summary, train_step)

                validation_num += len(batch_y)
                validation_loss += batch_loss
                validation_num_correct += num_correct

                validation_accuracy = validation_num_correct / float(validation_num)

            validation_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_accuracy',
                                                         simple_value=validation_accuracy)])
            validation_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_average_loss',
                                                                    simple_value=validation_loss/float(validation_num))])

            validation_writer.add_summary(validation_accuracy_summary, train_step)
            validation_writer.add_summary(validation_loss_summary, train_step)

            logger.info(
                "Learning Rate: {}, Epoch {}, {:.01f} sec| Loss (Train/Validation) : {:.04f}, {:.04f} | Accuracy (Train/Validation): {:.04f}, {:.04f}".format(
                    learning_rate, epoch, time.time() - epoch_time_start, training_loss, validation_loss,
                    train_accuracy,
                    validation_accuracy))

            if epoch % num_epochs_checkpoint == 0:
                model_save_path = saver.save(sess, log_dir, global_step=epoch)
                logger.info("Saved model at epoch {}: {}".format(epoch, str(model_save_path)))

        ########################################################################
        # Test
        ########################################################################
        test_num = 0
        test_num_correct = 0
        test_loss = 0

        for batch_x, batch_y in data.test_data_iterator(batch_size=batch_size, do_shuffle=False):
            num_correct, batch_loss, test_summary = sess.run([correct_prediction, loss,summary], {
                features_placeholder: batch_x,
                labels_placeholder: batch_y,
            })

            test_writer.add_summary(test_summary, train_step)

            test_num += len(batch_y)
            test_num_correct += num_correct
            test_loss += batch_loss

            test_accuracy = test_num_correct / float(test_num)

        test_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_accuracy',
                                                                         simple_value=test_accuracy)])
        test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='epoch_average_loss',
                                                                simple_value=test_loss/float(test_num))])

        test_writer.add_summary(test_accuracy_summary, train_step)
        test_writer.add_summary(test_loss_summary, train_step)

        logger.info("Test: Accuracy : {}. Test Loss {}".format(test_accuracy, test_loss))
        sess.close()

    duration = (time.time() - time_start) / 60.0
    logger.info("Completed training and validation in %.2f minutes." % duration)

    return training_loss, validation_loss, train_accuracy, validation_accuracy, model_save_path


# ==============================================================================
# Logging
# ==============================================================================

def setup_logger(log_dir):
    global logger

    log_level = getattr(logging, "INFO")
    log_filepath = os.path.join(log_dir, "model.log")

    logger.info("Given log_filepath = %s" % log_filepath)

    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format)
    logging.basicConfig(filename=log_filepath, level=log_level, format=format)

    # Print log to screen
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger()

    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logging.info('Log setup complete.')


def prepare_log_dir(log_dir):
    if tf.gfile.Exists(log_dir):
        print("Deleting old log directory.")
        tf.gfile.DeleteRecursively(log_dir)

    if not tf.gfile.Exists(log_dir):
        logger.info("Making log directory: %s" % log_dir)
        tf.gfile.MakeDirs(log_dir)


def get_name_for_model():
    name = ""
    for key, value in vars(FLAGS).items():
        name = name + str(key) + "." + str(value) + "_"

    return name


# ==============================================================================
# Get Parameters
# ==============================================================================

def convert_str_to_list(s, data_type):
    L = s.split(",")
    L = [data_type(z) for z in L]

    return L


# ==============================================================================
# Main
# ==============================================================================

def main(model_name,
         lr_initial,
         lr_decay_steps,
            model_type,
         num_epochs,
         batch_size,
         dropout_keep_prob,
         num_epochs_checkpoint,
         l2_penalty_multiplier,
         load_model_fullpath):
    """
    Main function to setup logs, and start training. This function can be run in parallel for grid search.
    """

    global log_dir

    root_output_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow.logs')
    log_dir = os.path.join(root_output_dir, model_name)

    prepare_log_dir(log_dir=log_dir)
    setup_logger(log_dir=log_dir)

    #If FLAGS != None, log it.
    logger.info("Command line parameters: {}".format(FLAGS))

    return run_training(log_dir=log_dir,
                        lr_initial=lr_initial,
                        lr_decay_steps=lr_decay_steps,
                        num_epochs=num_epochs,
                        model_type=model_type,
                        batch_size=batch_size,
                        dropout_keep_prob=dropout_keep_prob,
                        num_epochs_checkpoint=num_epochs_checkpoint,
                        l2_penalty_multiplier=l2_penalty_multiplier,
                        load_model_fullpath=load_model_fullpath)


# ==============================================================================
# __name__ == __main__
# ==============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Name of the model. This is also the name of the directory where log files are placed.'
    )

    parser.add_argument(
        '--lr_initial',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--lr_decay_steps',
        type=int,
        default=1200,
        help='Decay the learning rate after this many steps.'
    )

    parser.add_argument(
        '--load_model_fullpath',
        type=str,
        default=None,
        help='Fullpath of a saved model to load.'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3000,
        help='Number of epochs to run trainer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='inference',
        help='Model type. Can be inference or inference_LayerNormBasicLSTMCell'
    )

    parser.add_argument(
        '--dropout_keep_prob',
        type=str,
        default="0.5,0.5,0.5,1.0,1.0",
        help='Dropout applied to each layer, given as a list'
    )

    parser.add_argument(
        '--l2_penalty_multiplier',
        type=float,
        default=.001,
        help="Apply an L2 penalty with this multiplier to all trainable parameters."
    )

    FLAGS, unparsed = parser.parse_known_args()

    dropout_keep_prob = convert_str_to_list(FLAGS.dropout_keep_prob, data_type=float)

    model_name = FLAGS.model_name
    if model_name == None:
        model_name = get_name_for_model()

    main(model_name=model_name,
         lr_initial=FLAGS.lr_initial,
         lr_decay_steps=FLAGS.lr_decay_steps,
         model_type=FLAGS.model_type,
         num_epochs=FLAGS.num_epochs,
         batch_size=FLAGS.batch_size,
         dropout_keep_prob=dropout_keep_prob,
         num_epochs_checkpoint=NUM_EPOCHS_CHECKPOINT,
         l2_penalty_multiplier=FLAGS.l2_penalty_multiplier,
         load_model_fullpath=FLAGS.load_model_fullpath,
         )
