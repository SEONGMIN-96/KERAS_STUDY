# 실습 diabets
# 1. loss와 R2 평가를 함
# Minmax와 Standard 결과들 명시

from operator import mod
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, LSTM, Conv1D
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder

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
# print(x_train.shape, x_test.shape) 
# (379, 13) (127, 13)
# print(y_train.shape, y_test.shape)
# (379,) (127,)

x_train = x_train.reshape(379, 13, 1)
x_test = x_test.reshape(127, 13, 1)

# print('==============================')
# print(y_train.shape, y_test.shape)
# (379, 197) (127, 91)

# 2. 모델 구성

# model = Sequential()
# # model.add(LSTM(units=16, input_shape=(13, 1), activation='relu'))
# model.add(Conv1D(32, 2, input_shape=(13, 1)))
# model.add(LSTM(64, return_sequences=True))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))

model = load_model('./_save/ModelCheckPoint/keras48_1_boston_MCP.hdf5')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', mode='min', patience=30)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                        filepath='./_save/ModelCheckPoint/keras48_1_boston_MCP.hdf5')

model.compile(loss='mse', optimizer='adam', metrics='mae')

start_time = time.time()
# model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=8, shuffle=False, verbose=1, callbacks=[es, cp])
end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_1_boston_model_save.h5')

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss : ',loss[0])
print('mae : ',loss[1])
print('소요 시간 : ',end_time)

'''

after CNN

loss :  11.52690315246582
mae :  2.6335463523864746
소요 시간 :  17.769816398620605

after LSTM

loss :  19.389720916748047
mae :  3.566633701324463
소요 시간 :  81.20697975158691

after Conv1D

loss :  37.32260513305664
mae :  4.802786350250244
소요 시간 :  32.98319673538208

'''