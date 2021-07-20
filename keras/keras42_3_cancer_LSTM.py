from re import T
import numpy as np
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

# 1. 데이터

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) (569, 30) (569,)

# print(y[:20])
# print(np.unique(y))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True)

# print(x_train.shape, x_test.shape)
# (455, 30) (114, 30)

# 1_1. 데이터 전처리

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM

model = Sequential()
model.add(LSTM(32, input_shape=(30, 1), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=8, verbose=1, shuffle=True, validation_split=0.2, callbacks=es)
end_time = time.time() - start_time

# 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss는 :', loss[0])
print('accuracy는 :', loss[1])
print('소요 시간 : ', end_time)



'''

after CNN

loss는 : 0.1286865770816803
accuracy는 : 0.9473684430122375
소요 시간 :  26.262674808502197

after LSTM

loss는 : 0.1202988401055336
accuracy는 : 0.9473684430122375
소요 시간 :  360.1267182826996

'''