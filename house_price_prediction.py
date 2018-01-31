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