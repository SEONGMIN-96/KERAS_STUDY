# 앞뒤가 똑같은 오~토 인코더~~

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import binary_crossentropy

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

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_03 = autoencoder(hidden_layer_size=4)
model_04 = autoencoder(hidden_layer_size=8)
model_05 = autoencoder(hidden_layer_size=16)
model_06 = autoencoder(hidden_layer_size=32)

print("############## node 1개 시작 ##############")
model_01.compile(loss='binary_crossentropy', optimizer='adam')
model_01.fit(x_train, x_train, epochs=10)

print("############## node 2개 시작 ##############")
model_02.compile(loss='binary_crossentropy', optimizer='adam')
model_02.fit(x_train, x_train, epochs=10)

print("############## node 3개 시작 ##############")
model_03.compile(loss='binary_crossentropy', optimizer='adam')
model_03.fit(x_train, x_train, epochs=10)

print("############## node 4개 시작 ##############")
model_04.compile(loss='binary_crossentropy', optimizer='adam')
model_04.fit(x_train, x_train, epochs=10)

print("############## node 5개 시작 ##############")
model_05.compile(loss='binary_crossentropy', optimizer='adam')
model_05.fit(x_train, x_train, epochs=10)

print("############## node 개 시작 ##############")
model_06.compile(loss='binary_crossentropy', optimizer='adam')
model_06.fit(x_train, x_train, epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_03 = model_03.predict(x_test)
output_04 = model_04.predict(x_test)
output_05 = model_05.predict(x_test)
output_06 = model_06.predict(x_test)

fig, axes = plt.subplots(7, 5, figsize=(15, 15))

random_imgs = random.sample(range(output_01.shape[0]),5)
outputs = [x_test, output_01, output_02, output_03, output_04,
            output_05, output_06]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()