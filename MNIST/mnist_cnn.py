import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define inputs/outputs
x = tf.placeholder(tf.float32, [None, 784])

def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

def conv2d(x, y):
    return tf.nn.conv2d(x, y, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

w1_conv = weight_variables([5, 5, 1, 32])
b1_conv = bias_variables([32])

h1_conv = conv2d(x_image, w1_conv)
h1_pool = max_pool_2x2(h1_conv)

w2_conv = weight_variables([5, 5, 32, 64])
b2_conv = bias_variables([64])

h2_conv = conv2d(h1_pool, w2_conv)
h2_pool = max_pool_2x2(h2_conv)

h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])

w1_fc = weight_variables([7*7*64, 1024])
b1_fc = bias_variables([1024])

h1_fc = tf.nn.relu(tf.matmul(h2_pool_flat, w1_fc) + b1_fc)

keep_prob = tf.placeholder(tf.float32)

h1_fc_drop = tf.nn.dropout(h1_fc, keep_prob)

w2_fc = weight_variables([1024, 10])
b2_fc = bias_variables([10])

y_conv = tf.matmul(h1_fc_drop, w2_fc) + b2_fc

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(20000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    if i % 200 == 0:
        print("Epoch %d has passed."%i)

correct_predictions = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

total_sum = 0
for i in range(200):
    batch = mnist.train.next_batch(50)
    temp_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    total_sum += temp_accuracy

test_accuracy = total_sum / 200
print("test accuracy is %f"%test_accuracy)
