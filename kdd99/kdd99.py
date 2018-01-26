"""Builds the network.

Implements the inference/loss/training pattern for model building.

Credits: https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/mnist/mnist.py

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from helper_functions import *


# ==============================================================================
# Inference
# ==============================================================================

def inference(input_features):
    """The inference() function builds the graph as far as needed
     to return the tensor that would contain the output predictions.

    Args:
        input_features: placeholder for

    Returns:
        softmax_linear: Output tensor with the computed logits.

    """
    n_hidden_1 = 100

    # Linear
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.random_normal([num_features, n_hidden_1]), name='weights')
        biases = tf.Variable(tf.random_normal([n_hidden_1]), name='biases')
        linear = tf.add(tf.matmul(input_features, weights), biases, name='output')
        layer_1_output = tf.nn.relu(linear, "activation")

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([n_hidden_1, num_classes]), name='weights')
        biases = tf.Variable(tf.zeros([num_classes]), name='biases')
        layer_2_output = tf.add(tf.matmul(layer_1_output, weights), biases, name='output')

    return layer_2_output


# ==============================================================================
# Loss
# ==============================================================================

def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')

    # Mean of the cross entropy.
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss


# ==============================================================================
# Training
# ==============================================================================

def training(loss, learning_rate):
    """Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to
    `sess.run(...)` which causes the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """

    # Create the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Global variable to track number of training steps
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # This operation is what must be run in a TensorFlow session in order to induce one full step of training.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


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
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    correct_prediction = tf.reduce_sum(correct_prediction)

    return correct_prediction
