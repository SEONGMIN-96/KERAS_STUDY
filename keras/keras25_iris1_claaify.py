import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

# 1. 데이터

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

print(x)
print(y)

# 1_0. 원 핫 인코딩 ONE-HOT-ENCODING
# 0-> [1,0,0]
# 1-> [0,1,0]
# 2-> [0,0,1]

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
print('yddd',y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True)

# 1_1. 데이터 전처리

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, verbose=1, shuffle=True, validation_split=0.2, callbacks=es)

# 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss는 :', loss[0])
print('accuracy는 :', loss[1])