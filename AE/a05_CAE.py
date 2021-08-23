# 2번 카피해서 복붙
# CNN으로 딥하게 구성
# 2개의 모델을 구성하는데 하나는 기본적 오토인코더
# 다른 하나는 딥하게 만든 구성
# 2개의 성능 비교

# 앞뒤가 똑같은 오~토 인코더~~

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

bx_train = x_train.reshape(60000, 28*28).astype('float')/255
bx_test = x_test.reshape(10000, 28*28).astype('float')/255

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255

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

model = autoencoder(hidden_layer_size=154)      #pca 95%

model.compile(loss='mse', optimizer='adam')
model.fit(bx_train, bx_train, epochs=30, batch_size=256)

output = model.predict(bx_test)

def autoencoder_deep(hidden_layer_size):
    model01 = Sequential()
    model01.add(Conv2D(filters=hidden_layer_size, kernel_size=(1,1), input_shape=(28, 28, 1)))
    model01.add(MaxPool2D())
    model01.add(Conv2D(filters=512, kernel_size=(1,1), activation='relu'))
    model01.add(UpSampling2D())
    model01.add(Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid'))
    return model01

model01 = autoencoder_deep(hidden_layer_size=154)      #pca 95%

model01.summary()

model01.compile(loss='mse', optimizer='adam')
model01.fit(x_train, x_train, epochs=30, batch_size=256)

output1 = model01.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.

random_images = random.sample(range(output.shape[0]),5)

# 오토 인코더 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# cnn_deep
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()