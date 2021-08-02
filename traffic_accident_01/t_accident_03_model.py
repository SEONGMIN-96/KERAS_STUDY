from datetime import date
import pandas as pd
import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPool1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.ops.gen_math_ops import Max


# 1. 데이터

# picle 
 
np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

x_data = np.load('./_save/_npy/t_accident_x_data.npy').astype(float)
y_data = np.load('./_save/_npy/t_accident_y_data.npy').astype(float)

print(x_data.shape)
print(y_data.shape)

x_data = x_data.reshape(4018, 4, 1)

# train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape)
print(x_test.shape)

# 2. 모델구성

model = Sequential()
model.add(LSTM(64, input_shape=(4, 1), return_sequences=True))
model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
model.add(Conv1D(128, 2, activation='relu', padding='same'))
model.add(MaxPool1D())
model.add(Conv1D(128, 2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(512, activation='same'))
model.add(Dense(3))

# 3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=500, batch_size=16, verbose=1, validation_split=0.2,
            callbacks=[es])

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)s

print('val_loss : ',loss[0])
print('val_acc : ',loss[1])
