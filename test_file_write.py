import tensorflow as tf
import numpy as np

W = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run(W)
file = open("parameters.npy", "wb")
np.save(file, a)
file.close()

file = open("parameters.npy", "rb")
b = np.load(file)

print  b

#
# def read_file(file_name):
#     file = open(file_name)
#     content_list = file.readlines()
#     # print content_list
#     str = ''
#     for element in content_list:
#         str += element
#     file.close()
#     # print eval(str)
#     return tf.constant(eval(str))
#
# sess = tf.Session()
#
# read_file("file_out.txt")
#
# a = sess.run(read_file("file_out.txt"))
# print a
#
# file = open("file_out.txt", "w")
# print >> file, a


# backup
# file = open("b_fc2.txt", "w")
# print >> file, sess.run(b_fc2)
# file.close()