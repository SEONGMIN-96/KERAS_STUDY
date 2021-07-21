# 실습 diabets
# 1. loss와 R2 평가를 함
# Minmax와 Standard 결과들 명시

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
# scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# print(x[:1])
# print(y[:1])
# print(x.shape, y.shape) # (442, 10) (442,)

# print(x_train.shape, x_test.shape)
# (331, 10) (111, 10)

x_train = x_train.reshape(331, 10, 1)
x_test = x_test.reshape(111, 10, 1)

# print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# print(datasets.DESCR)
# :Attribute Information:
#       - age     age in years - 나이
#       - sex     - 성별
#       - bmi     body mass index - 체질량 지수
#       - bp      average blood pressure - 평균 혈압
#       - s1      tc, T-Cells (a type of white blood cells)
#       - s2      ldl, low-density lipoproteins
#       - s3      hdl, high-density lipoproteins
#       - s4      tch, thyroid stimulating hormone
#       - s5      ltg, lamotrigine
#       - s6      glu, blood sugar level

# print(y[:30])
# print(np.min(y), np.max(y))

# 2. 모델 구성

model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(10, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))



model.summary()

# model.save('./_save/keras46_1_save_model_1.h5')
# model.save_weights('./_save/keras46_1_save_weight_1.h5')

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')

model.load_weights('./_save/keras46_1_save_weight_1.h5')
# model.load_weights('./_save/keras46_1_save_weight_2.h5')

import time

start_time = time.time()
# model.fit(x_train, y_train, epochs=50, batch_size=8, shuffle=False, verbose=1)
end_time = time.time() - start_time

# model.save('./_save/keras46_1_save_model_4.h5')
# model.save_weights('./_save/keras46_1_save_weight_2.h5')


# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

print('loss : ',loss[0])
print('mae : ',loss[1])
print('소요 시간 : ',end_time)

'''

after CNN

loss :  6657.14013671875
mae :  62.17472457885742
소요 시간 :  65.1311707496643

after LSTM

loss :  24.36919403076172
mae :  3.6350646018981934
소요 시간 :  127.2862548828125

'''