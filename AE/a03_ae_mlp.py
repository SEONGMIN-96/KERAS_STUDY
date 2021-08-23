# 앞뒤가 똑같은 오~토 인코더~~
# 딥하게 구성

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float')/255
x_test = x_test.reshape(10000, 28*28).astype('float')/255

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
import random

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),  
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder_deep(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),  
                    activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)      #pca 95%
model1 = autoencoder_deep(hidden_layer_size=154)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, x_train, epochs=30, batch_size=16)

model1.compile(loss='mse', optimizer='adam')
model1.fit(x_train, x_train, epochs=30, batch_size=16)

output = model.predict(x_test)
output1 = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.

random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# deep_model 출력
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("DEEP", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()