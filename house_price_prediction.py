"""
    Simple predction of house prices based on the size of house
"""
import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generation of random houses
# generation of house size
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house price
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000,
    high=70000, size=num_house)

# plot generated
plt.plot(house_size, house_price, "bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

# prepare the data
def normalize(array):
    return (array - array.mean())/array.std()

# train test split
num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# putting data in tensor containers
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# tensor variables
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# inference operation
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size),
     tf_price_offset)

# loss function
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2))/(2*num_train_samples)

learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

# launch graph in the session
with tf.Session() as sess:
    sess.run(init)

    # set how often to display training progress and iterations
    display_every = 2
    num_training_iter = 50

    # keep iterating the data
    for iteration in range(num_training_iter):

        # fit all training data
        for (x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x,
                                            tf_price: y})

        # display current status
        if(iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, 
            feed_dict={tf_house_size:train_house_size_norm,
                        tf_price:train_price_norm})
            print("iteration #:", '%04d' % (iteration+1),
                "cost=", "{:.9f}".format(c),
                "size_factor=", sess.run(tf_size_factor),
                "price_offset=", sess.run(tf_price_offset))

    print("Optimisation Finished")
    training_cost = sess.run(tf_cost, 
            feed_dict={tf_house_size: train_house_size_norm,
                    tf_price:train_price_norm})
    print("Trained cost=", training_cost,
     "size_factor=", sess.run(tf_size_factor),
     "price_offset=", sess.run(tf_price_offset), '\n')

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

     # plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training Data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing Data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
     (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
     label = 'Learned Regression')

    plt.legend(loc="upper left")
    plt.show()


     