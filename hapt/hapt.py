"""Builds the network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is not meant to be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import input_data
import tensorflow as tf

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

NUM_FEATURES = input_data.NUM_FEATURES
NUM_CLASSES = input_data.NUM_CLASSES

LAYER_SIZES = [32,32]
n_hidden_1 = 32

# ==============================================================================
# Inference
# ==============================================================================

def inference(input_features, dropout_keep_prob):
    """The inference() function builds the graph as far as needed
     to return the tensor that would contain the output predictions.

    Args:
        rows: rows placeholder, from inputs().

    Returns:
        softmax_linear: Output tensor with the computed logits.

    """
    rnn_layers = []
    for i, size in enumerate(LAYER_SIZES):
        cell = tf.nn.rnn_cell.BasicLSTMCell(size)

        # Add Dropout
        if (dropout_keep_prob != None):
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob[i])

        rnn_layers.append(cell)

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                inputs=input_features,
                                                dtype=tf.float32)

    tf.summary.histogram('rnn_output', rnn_outputs[-1])

    num_rnn_outputs = LAYER_SIZES[-1]

    feedforward_hidden_1 = tf.contrib.layers.fully_connected(inputs=rnn_outputs[-1],
                                                             num_outputs=n_hidden_1,
                                                             scope='feedforward_hidden1')

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([n_hidden_1, NUM_CLASSES]), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        matmul = tf.matmul(feedforward_hidden_1, weights, name='matmul')
        logits = tf.add(matmul, biases, name='output')

    return logits


def inference_LayerNormBasicLSTMCell(input_features, dropout_keep_prob):
    """The inference() function builds the graph as far as needed
     to return the tensor that would contain the output predictions.

    Args:
        rows: rows placeholder, from inputs().

    Returns:
        softmax_linear: Output tensor with the computed logits.

    """
    rnn_layers = []
    for i, size in enumerate(LAYER_SIZES):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(size, dropout_keep_prob=dropout_keep_prob[i])

        rnn_layers.append(cell)

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                inputs=input_features,
                                                dtype=tf.float32)

    num_rnn_outputs = LAYER_SIZES[-1]

    # Feedforward Layers
    feedforward_hidden_1 = tf.contrib.layers.fully_connected(inputs=rnn_outputs[-1],
                                                             num_outputs=n_hidden_1,
                                                             scope='feedforward_hidden1')

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([n_hidden_1, NUM_CLASSES]), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        matmul = tf.matmul(feedforward_hidden_1, weights, name='matmul')
        logits = tf.add(matmul, biases, name='output')

    return logits

def inference_benchmark(input_features, dropout_keep_prob):
    """
    This inference function is simple logistic regression. Determine the speed difference
    between the inference function above, and simple logistic regression. If the difference is minimal, then
    the input pipeline is likely the bottleneck for speed.
    :param input_features:
    :return:
    """
    """
    :param input_features: 
    :return: 
    """

    print("*" * 100)
    print("This inference function is used for BENCHMARKING only.")
    print("*" * 100)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], dtype=tf.float32), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES], dtype=tf.float32), name='biases')
        matmul = tf.matmul(input_features, weights, name='matmul')
        logits = tf.add(matmul, biases, name='output')

    return logits


# ==============================================================================
# Loss
# ==============================================================================

def loss(logits, labels, l2_penalty_multiplier=.001):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """

    l2_norm = tf.reduce_sum(
        [tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()])
    l2_penalty = l2_penalty_multiplier * l2_norm
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    cross_entropy = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    loss = cross_entropy + l2_penalty

    tf.summary.scalar('l2_norm_trainable_parameters', l2_norm)
    tf.summary.scalar('l2_penalty', l2_penalty)

    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('loss', loss)

    return loss


# ==============================================================================
# Training
# ==============================================================================

def training(loss, lr_initial, lr_decay_steps):
    """The training() function adds the operations needed to minimize the loss via Gradient Descent.

    1.) Creates a summarizer to track the loss over time in TensorBoard.

    2.) Creates an optimizer and applies the gradients to all trainable variables.

    The Op train_op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        :return train_op: The Op for training.
        :return learning_rate: Tensor containing current learning rate
        :return global_step: Global step number.
    """
    # Global variable to track number of training steps
    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(lr_initial, global_step,
                                               lr_decay_steps, 0.9, staircase=True)

    # boundaries = [1000, 3000]
    # values = [.01, 0.066, 0.033]
    # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    tf.summary.scalar('learning_rate', learning_rate)

    # Step 1: Create the optimizer
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # This operation is what must be run in a TensorFlow session in order to induce one full step of training.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return learning_rate, train_op, global_step


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar with the number of examples that were predicted correctly.
    """

    prediction = tf.nn.softmax(logits)

    # Number of correct predictions
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    num_correct = tf.reduce_sum(correct_prediction)
    tf.summary.scalar('num_correct', num_correct)

    # Accuracy
    accuracy = 100 * tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    return num_correct
