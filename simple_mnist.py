# Simple MINST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Helper function to download the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# place holder for the 28 x 28 (784) image
x = tf.placeholder(tf.float32, shape=[None, 784])

# 10 element vector - probability of each number that the image represents
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# defining weights and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define model
# multiply each value by weight and sum up product
# and apply to an activation function - softmax
y = tf.nn.softmax(tf.matmul(x,W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

# reduce the loss using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialise global varibales
init = tf.global_variables_initializer()

# create a session
sess = tf.Session()

# perform the initialization which is only the initialization of all global variables
sess.run(init)

# perform 1000 training steps
for i in range(1000):
    # get 100 random data points from the data
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # optimisation with the data
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# evalauate how well the model performed
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# cast the true-false to number
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run the trained model on test data
test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()


