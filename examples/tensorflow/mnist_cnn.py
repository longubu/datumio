# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A very simple MNIST classifier.


See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md

Adapted for use of datumio (although very trivial in this case)
Modified to include CNN tutorial.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import datumio.datagen as dtd
import tensorflow as tf


def weight_variable(shape):
    """Initialize weights given an input shape (iterable)"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initialize parameters of bias given an input shape (iterable)"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """Create a convolution layer, given inputs x and
    weights W (tf.Variable)"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """Create a max_pool layer, given inputs x"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# get mnist data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
n_epochs = 400

# create batch generator with random augmentations applied on-the-fly
rng_aug_params = {'rotation_range': (-20, 20),
                  'translation_range': (-4, 4),
                  'do_flip_lr': True}
batch_generator = dtd.BatchGenerator(
                     mnist.train._images.reshape(-1, 28, 28, 1),
                     y=mnist.train._labels, rng_aug_params=rng_aug_params)

# start tensorflow interactive session
sess = tf.InteractiveSession()

# ----- create the model -----
# define inputs
print("Generating tensorflow CNN model")
# x = tf.placeholder(tf.float32, [None, 784])
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer (softmax classification layer)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start session
sess.run(tf.initialize_all_variables())

# train network
print("Starting to train model")
for i in range(n_epochs):
    for batch in batch_generator.get_batch(batch_size=50, shuffle=True):
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("... epoch %d/%d, training accuracy %g"
          % (i + 1, n_epochs, train_accuracy))

# test network
print("# -------------- #")
print("Test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images.reshape(-1, 28, 28, 1), y_: mnist.test.labels,
    keep_prob: 1.0}))
