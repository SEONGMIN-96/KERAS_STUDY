import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers.recurrent import LSTM

# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
model.add(Conv1D(64, 2, input_shape=(2, 392)))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64, shuffle=True, validation_split=0.2,
                    callbacks=es)
end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss는 :', loss[0])
print('acc는 :', loss[1])
print('걸린시간(분) : ', end_time / 60)

'''

loss는 : 0.09243981540203094
acc는 : 0.9829999804496765

loss는 : 0.082391157746315
acc는 : 0.9830999970436096

loss는 : 0.06445126980543137
acc는 : 0.9901999831199646

loss는 : 0.07425221055746078
acc는 : 0.9914000034332275

after DNN

loss는 : 0.19932815432548523
acc는 : 0.9613000154495239

after Dropout

loss는 : 0.30774420499801636
acc는 : 0.9200999736785889
걸린시간(분) :  6.7394913037618

after LSTM

loss는 : 2.2807400226593018
acc는 : 0.12929999828338623
걸린시간(분) :  33.237756768862404

after Conv1D

loss는 : 0.2598702013492584
acc는 : 0.9337999820709229
걸린시간(분) :  4.845967352390289

'''