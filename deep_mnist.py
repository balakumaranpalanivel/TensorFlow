# Deep MNIST with convolutional neural network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Helper function to download the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# using interactive session makes it default 
sess = tf.InteractiveSession()

# place holder for the 28 x 28 (784) image
x = tf.placeholder(tf.float32, shape=[None, 784])

# 10 element vector - probability of each number that the image represents
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# reshape the image to preserve the 2D shape of the image
x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

# define helper methods to create weight and bias

# since activation function - relu, we have to initialise the weight and 
# bias to small positive value
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# implement convolutional and pooling layers
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
    strides=[1,2,2,1], padding='SAME')

# defining the layers in Neural Network

# 1st Convolutional layer
# 32 features for each 5x5 patch of the image
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# Do the convolution using activation function relu
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# the results are passed over to a max_pool layer
h_pool1 = max_pool_2x2(h_conv1)

# 2nd Convolutional layer
# Process the 32 features from previous layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# do convolution using previous layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# the results are max pooled
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# convert the data into a flat image for fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# to prevent overfitting of trainig data - add dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define the model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_conv, labels=y_
))

# Loss function - ADAM optimiser
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy Computation
# determine correct predictions
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# determine accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialise and start the session
sess.run(tf.global_variables_initializer())

# Train the model
import time

# define the number of epochs
num_steps = 3000
display_every =  100

# Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    # periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0}) 
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))

# Display summary 
#     Time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

#     Accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

sess.close()






