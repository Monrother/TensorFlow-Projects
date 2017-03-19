import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define inputs/outputs
x = tf.placeholder(tf.float32, [None, 784])

def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, y):
    return tf.nn.conv2d(x, y, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variables([5, 5, 1, 32])
b_conv1 = bias_variables([32])

h1_conv = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h1_pool = max_pool_2x2(h1_conv)

W_conv2 = weight_variables([5, 5, 32, 64])
b_conv2 = bias_variables([64])

h2_conv = tf.nn.relu(conv2d(h1_pool, W_conv2) + b_conv2)
h2_pool = max_pool_2x2(h2_conv)

h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])

W_fc1 = weight_variables([7 * 7 * 64, 1024])
b_fc1 = bias_variables([1024])

h1_fc = tf.nn.relu(tf.matmul(h2_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

h1_fc_drop = tf.nn.dropout(h1_fc, keep_prob)

W_fc2 = weight_variables([1024, 10])
b_fc2 = bias_variables([10])

y_conv = tf.matmul(h1_fc_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_predictions = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# when epoch=20000 the accuracy is only 10%, something wrong, so make epochs a parameter to test

for i in range(20000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 200 == 0:
        print("Epoch %d has passed."%i)

# evaluate the mean accuracy of the test set by evaluating 200 test examples 200 times
total_sum = 0
for i in range(200):
    batch = mnist.train.next_batch(200)
    temp_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    total_sum += temp_accuracy

test_accuracy = total_sum / 200
print("test accuracy is %f"%test_accuracy)

# np.set_printoptions(threshold="nan")

# save all the trained variables to file
def save_params(file_name, variable):
    file = open(file_name, "wb")
    np.save(file, sess.run(variable))
    file.close()

save_params("W_conv1.npy", W_conv1)
save_params("W_conv2.npy", W_conv2)
save_params("b_conv1.npy", b_conv1)
save_params("b_conv2.npy", b_conv2)
save_params("W_fc1.npy", W_fc1)
save_params("W_fc2.npy", W_fc2)
save_params("b_fc1.npy", b_fc1)
save_params("b_fc2.npy", b_fc2)

