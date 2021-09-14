import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. 데이터

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6 ,7])

print(x.shape, y.shape)  # (4, 3) (4,)

x = x.reshape(4, 3, 1)   # (batch_size, timesteps, feature)

# 2. 모델 구성

model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(32, activation='relu'))     # timesteps       feature
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

'''

    # stackoverflow

num_units = equals the number of units in the RNN

num_features = equals the number features of your input

recurrent_weights = num_units*num_units

input_weights = num_features*num_units

biases = num_units*1

recurrent_weights + input_weights + biases

or

num_units* num_units + num_features* num_units + biases

= (num_features + num_units)* num_units + biases

10*10 + 1*10 + 10 = 120

'''
'''

    # 선생님

(Input + bias) * output + output * output
= (Input + bias + output) * output

'''


# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=30)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2, callbacks=es)

# 4. 평가, 예측

x_predict = np.array([[5], [6], [7]]).reshape(1, 3, 1)

predict = model.predict(x_predict)

print(predict) # [[8.697411]], [[8.287933]], [[8.038916]]