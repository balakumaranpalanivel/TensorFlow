# import TensorFlow
import tensorflow as tf

sess = tf.Session()

# verify we can print a string
hello = tf.constant("Hello World!")
print(sess.run(hello))

# simple math
a = tf.constant(20)
b = tf.constant(22)
print("Sum: {0}".format(sess.run(a+b)))