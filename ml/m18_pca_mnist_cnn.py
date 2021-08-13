# 실습
# mnist 데이터를 pca를 통해 cnn으로 구성
# 
# (28, 28) -> 784 -> 차원축소 (400) -> (20, 20) -> CNN 모델구성

import numpy as np
from numpy.lib.format import open_memmap
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
import warnings
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPool2D, Flatten, Dropout
import time

warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

# reshape

x = x.reshape(70000, 28*28)

# pca

pca = PCA(n_components=20*20)

x = pca.fit_transform(x)

x = x.reshape(70000, 20, 20, 1)

# train_test

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66
)

# model 

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(20, 20, 1)))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# compile, fit

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

# evaluate

loss, acc = model.evaluate(x_test, y_test)

print("loss :", loss)
print("acc :", acc)

'''
loss : 27.538854598999023
acc : 0.09799999743700027
'''
