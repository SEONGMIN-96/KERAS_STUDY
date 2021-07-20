import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

# 1. 데이터

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6 ,7])

print(x.shape, y.shape)  # (4, 3) (4,)

x = x.reshape(4, 3, 1)   # (batch_size, timesteps, feature)

# 2. 모델 구성

model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
# model.add(LSTM(units=10, activation='relu', input_shape=(3, 1)))
model.add(GRU(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=30)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2, callbacks=es)

# 4. 평가, 예측

x_predict = np.array([[5], [6], [7]]).reshape(1, 3, 1)

predict = model.predict(x_predict)

print(predict) 

'''

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390
_________________________________________________________________
dense (Dense)                (None, 32)                352
_________________________________________________________________
dense_1 (Dense)              (None, 64)                2112
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160
_________________________________________________________________
dense_3 (Dense)              (None, 128)               8320
_________________________________________________________________
dense_4 (Dense)              (None, 64)                8256
_________________________________________________________________
dense_5 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_6 (Dense)              (None, 16)                528
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 17
=================================================================
Total params: 26,215
Trainable params: 26,215
Non-trainable params: 0

***************************************************************
*               param = 390                                   *
* -> 3 * (Input + bias + output + reset_gate) * output = 390  *
***************************************************************

'''
