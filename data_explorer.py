import keras.datasets.mnist as mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check images


plt.imshow(x_train[0], cmap='Greys')
plt.show()