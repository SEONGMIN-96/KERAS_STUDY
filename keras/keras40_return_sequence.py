import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, MaxPool1D, GRU

# 1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
                [5,6,7], [6,7,8], [7,8,9], [8,9,10],
                [9,10,11], [10,11,12],
                [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape)  # (13, 3) (13,)

x = x.reshape(13, 3, 1)   # (batch_size, timesteps, feature)
# x_predict = x_predict.reshape(1, 3, 1)

x_predict = x_predict.reshape(1, x.shape[1], 1)

# 2. 모델 구성

model = Sequential()
# model.add(SimpleRNN(units=64, activation='relu', input_shape=(3, 1)))
model.add(LSTM(units=32, activation='relu', input_shape=(3, 1), return_sequences=True))
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1, shuffle=True)

# 4. 평가, 예측

predict = model.predict(x_predict)

print(predict) 

'''

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 7)                 504
_________________________________________________________________
dense (Dense)                (None, 5)                 40
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
=================================================================
Total params: 1,030
Trainable params: 1,030
Non-trainable params: 0

'''

