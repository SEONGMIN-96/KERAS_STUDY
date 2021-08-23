# 실습
# conv2D

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train_dnn = x_train.reshape(60000, 28*28).astype('float')/255
x_test_dnn = x_test.reshape(10000, 28*28).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

x_train_cnn = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_test_cnn = x_test.reshape(10000, 28, 28, 1).astype('float')/255

x_train_cnn_noised = x_train_cnn + np.random.normal(0, 0.1, size=x_train_cnn.shape)
x_test_cnn_noised = x_test_cnn + np.random.normal(0, 0.1, size=x_test_cnn.shape)
x_train_cnn_noised = np.clip(x_train_cnn_noised, a_min=0, a_max=1)
x_test_cnn_noised = np.clip(x_test_cnn_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPool2D, UpSampling2D
import matplotlib.pyplot as plt
import random

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),  
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder_deep(hidden_layer_size):
    model01 = Sequential()
    model01.add(Conv2D(filters=hidden_layer_size, kernel_size=(1,1), input_shape=(28, 28, 1)))
    model01.add(MaxPool2D())
    model01.add(Conv2D(filters=512, kernel_size=(1,1), activation='relu'))
    model01.add(UpSampling2D())
    model01.add(Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid'))
    return model01

'''
model = autoencoder(hidden_layer_size=154)      #pca 95%

model.compile(loss='mse', optimizer='adam')
model.fit(x_train_noised, x_train, epochs=30, batch_size=128)

output = model.predict(x_test)
'''

model = autoencoder_deep(hidden_layer_size=154)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train_cnn_noised, x_train_cnn, epochs=30, batch_size=128)

output = model.predict(x_test_cnn_noised)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(2  , 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.

random_images = random.sample(range(output.shape[0]),5)

# 원본
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# noise cnn
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE_CNN", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
