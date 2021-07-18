from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(x_test.shape, y_test.shape)
# (10000, 32, 32, 3) (10000, 1)

plt.imshow(x_train[40])
plt.show()