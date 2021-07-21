import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

# 1. 데이터

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

print(x)
print(y)

# 1_0. 원 핫 인코딩 ONE-HOT-ENCODING
# 0-> [1,0,0]
# 1-> [0,1,0]
# 2-> [0,0,1]

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True)

# 1_1. 데이터 전처리

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# print(x_train.shape, x_test.shape)
# (120, 4) (30, 4)

x_train = x_train.reshape(120, 2, 2)
x_test = x_test.reshape(30, 2, 2)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, Conv1D

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(2, 2)))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=8, verbose=1, shuffle=True, validation_split=0.2, callbacks=es)
end_time = time.time() - start_time

# 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('accuracy :', loss[1])
print('소요 시간 : ', end_time)


'''

after CNN

loss : 0.024418456479907036
accuracy : 1.0
소요 시간 :  5.661717653274536

after LSTM

loss : 0.15770858526229858
accuracy : 0.8999999761581421
소요 시간 :  4.950854539871216

after Conv1D

loss : 0.10457682609558105
accuracy : 0.9333333373069763
소요 시간 :  6.805733919143677

'''