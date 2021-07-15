# 실습 diabets
# 1. loss와 R2 평가를 함
# Minmax와 Standard 결과들 명시

from operator import mod
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# print(x[:1])
# print(y[:1])
# print(x.shape, y.shape) # (442, 10) (442,)

# 2. 모델 구성

input1 = Input(shape=(13,))
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

from tensorflow.keras.callbacks import EarlyStopping

# 3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', mode='min', patience=30)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=8, shuffle=False, verbose=2, callbacks=es)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)


# mse, R2

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('loss는 :', loss)
print('r2는 :', r2)

'''

MaxAbsScaler 후

loss는 : [3305.5283203125, 46.72243881225586]
r2는 : 0.5001717729908288

loss는 : [4531.357421875, 51.07051467895508]
r2는 : 0.3148144687704548

RobustScale 후

loss는 : [3684.0869140625, 48.10907745361328]
r2는 : 0.4429300629289523


QuantileTransformer 후

loss는 : [3573.439453125, 48.8539924621582]
r2는 : 0.4596610058932964

loss는 : [3342.4091796875, 46.705101013183594]
r2는 : 0.4945950524036312

PowerTransformer 후

loss는 : [3732.193359375, 49.189918518066406]
r2는 : 0.43565587335720923

loss는 : [3356.868408203125, 47.63866424560547]
r2는 : 0.4924086454532388

'''