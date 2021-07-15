# 실습 diabets
# 1. loss와 R2 평가를 함
# Minmax와 Standard 결과들 명시

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape)

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

input1 = Input(shape=(10,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)
dense5 = Dense(8, activation='relu')(dense4)
dense6 = Dense(4, activation='relu')(dense5)
dense7 = Dense(2, activation='relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)
model.summary()

# model = Sequential()
# model.add(Dense(96, input_shape=(10,), activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))  
# model.add(Dense(4, activation='relu'))           
# model.add(Dense(2, activation='relu'))          #활성화 함수
# model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=300, batch_size=8, shuffle=False, verbose=2)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss는 :', loss)

# mse, R2

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2는 :', r2)

'''

MaxAbsScaler 후

loss는 : [3456.0771484375, 48.432003021240234]
r2는 : 0.4774073298388566

loss는 : [3311.15771484375, 46.98512649536133]
r2는 : 0.49932054506993606

RobustScaler 후

loss는 : [3381.334716796875, 48.15561294555664]
r2는 : 0.4887091201002969

QuantileTransformer 후

loss는 : [3289.29443359375, 46.533451080322266]
r2는 : 0.5026264921987758

PowerTransformer 후

loss는 : [3298.06103515625, 46.6485595703125]
r2는 : 0.5013008889244287

loss는 : [3311.2529296875, 47.02490997314453]
r2는 : 0.49930623819796494

'''