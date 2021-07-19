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

x_train = x_train.reshape(3428, 11, 1, 1)
x_test = x_test.reshape(1470, 11, 1, 1)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(11, 1, 1), padding='same'))
model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))

# input1 = Input(shape=(11,))
# dense1 = Dense(128, activation='relu')(input1)
# dense2 = Dense(64, activation='relu')(dense1)
# dense3 = Dense(64, activation='relu')(dense2)
# dense4 = Dense(32, activation='relu')(dense3)
# dense5 = Dense(32, activation='relu')(dense4)
# output1 = Dense(16, activation='relu')(dense5)

# input2 = Input(shape=(11,))
# dense11 = Dense(64, activation='relu')(input2)
# dense12 = Dense(32, activation='relu')(dense11)
# dense13 = Dense(32, activation='relu')(dense12)
# dense14 = Dense(32, activation='relu')(dense13)
# dense15 = Dense(32, activation='relu')(dense14)
# output2 = Dense(16, activation='relu')(dense15)

# from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output2])
# merge2 = Dense(16)(merge1)
# merge3 = Dense(16, activation='relu')(merge2)

# last_output = Dense(7, activation='softmax')(merge3)

# model = Model(inputs=[input1, input2], outputs=last_output)


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

'''