import numpy as np

filepath='./_save/_npy/'

x_train = np.load(filepath+'k55_x_train_mnist.npy')
y_train = np.load(filepath+'k55_y_train_mnist.npy')
x_test = np.load(filepath+'k55_x_test_mnist.npy')
y_test = np.load(filepath+'k55_y_test_mnist.npy')

import matplotlib.pyplot as plt

# 1. 데이터

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 1_1. preprocessing

x_train = x_train.reshape(60000, 2, 392)
x_test = x_test.reshape(10000, 2, 392)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# y = np.reshape(y,(1,-1))
# print(y.shape)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.fit_transform(y_test).toarray()

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, LSTM, Conv1D

model = Sequential()
model.add(Conv1D(32, 2, input_shape=(2, 392)))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, shuffle=True, validation_split=0.2,
                    callbacks=es)
end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss는 :', loss[0])
print('acc는 :', loss[1])
print('걸린시간(분) : ', end_time / 60)

'''



'''