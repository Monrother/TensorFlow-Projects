import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define inputs/outputs
x = tf.placeholder(tf.float32, [None, 784])

def conv2d(x, y):
    return tf.nn.conv2d(x, y, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def read_file(file_name):
    file = open(file_name, "rb")
    return tf.constant(np.load(file))

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = read_file("W_conv1.npy")
b_conv1 = read_file("b_conv1.npy")

h1_conv = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h1_pool = max_pool_2x2(h1_conv)

W_conv2 = read_file("W_conv2.npy")
b_conv2 = read_file("b_conv2.npy")

h2_conv = tf.nn.relu(conv2d(h1_pool, W_conv2) + b_conv2)
h2_pool = max_pool_2x2(h2_conv)

h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])

W_fc1 = read_file("W_fc1.npy")
b_fc1 = read_file("b_fc1.npy")

h1_fc = tf.nn.relu(tf.matmul(h2_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

h1_fc_drop = tf.nn.dropout(h1_fc, keep_prob)

W_fc2 = read_file("W_fc2.npy")
b_fc2 = read_file("b_fc2.npy")

y_conv = tf.matmul(h1_fc_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

sess = tf.InteractiveSession()

correct_predictions = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# evaluate the mean accuracy of the test set by evaluating 200 test examples 200 times
total_sum = 0
for i in range(200):
    batch = mnist.train.next_batch(200)
    temp_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    total_sum += temp_accuracy

test_accuracy = total_sum / 200
print("test accuracy is %f"%test_accuracy)
