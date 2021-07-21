import numpy as np
import pandas as pd
from tensorflow.python.keras.layers.convolutional import Conv2D

# 1. 데이터

location = 'D:/_data/'
fname = 'winequality-white.csv'
datasets = pd.read_csv(location + fname, sep=';',
                        index_col=None, header=0)

# print(datasets.shape) (4898, 12)

# print(datasets.info())
# print(datasets.describe())

x = np.array(datasets.loc[:,'fixed acidity':'alcohol'])
y = np.array(datasets.loc[:,'quality'])

# print(x.shape) (4898, 11)
# print(y.shape) (4898,)

# print(np.min(y), np.max(y)) 3 9
# print(np.unique(y)) 7

# 1_0. One_Hot_Encoding

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# y = np.reshape(y,(1,-1))
# print(y.shape)

y = y[:,np.newaxis]

enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

# y = to_categorical(y)

# 1_1. train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                            train_size=0.7, shuffle=True, random_state=66)

# print(y_train.shape)

# 1_2. preprocessing

from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler

# scaler = PowerTransformer()
# scaler = StandardScaler()
# scaler = QuantileTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# print(x_train.shape, x_test.shape)
# (3428, 11) (1470, 11)

x_train = x_train.reshape(3428, 1, 11)
x_test = x_test.reshape(1470, 1, 11)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, Conv1D

model = Sequential()
model.add(Conv1D(64, 1, input_shape=(1, 11)))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=15)

model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics='acc')

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=1, validation_split=0.3, callbacks=es)
end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss :', loss[0])
print('acc :', loss[1])
print('소요 시간: ', end_time)

'''

after QuantileTransformer, to_categorical

loss는 : 3.7295846939086914
acc는 : 0.5867347121238708

after PowerTranfomer, after OneHotEncoder

loss는 : 2.2670493125915527
acc는 : 0.563265323638916

after validation_split = 0.3

loss는 : 1.0942986011505127
acc는 : 0.5265306234359741

after CNN

loss : 1.1293832063674927
acc : 0.5088435411453247
소요 시간:  22.23252582550049

after LSTM

loss : 1.107799768447876
acc : 0.5258503556251526
소요 시간:  94.81492042541504

after Conv1D

loss : 1.1335917711257935
acc : 0.48775508999824524
소요 시간:  60.44963765144348

loss : 1.1217396259307861
acc : 0.5074830055236816
소요 시간:  47.20397186279297

'''




