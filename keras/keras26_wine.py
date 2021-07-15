import numpy as np
from sklearn.datasets import load_wine

# 완성
# acc 0.8 이상!!

# 1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

# 1_1. 원핫인코딩!

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

# print(datasets.DESCR)
# print(datasets.feature_names)
# print(x.shape, y.shape) (178, 13) (178,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True)

# 1_2. preprocessing

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=30)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, callbacks=es, validation_split=0.2)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss는 :', loss[0])
print('acc는 :', loss[1])

'''
without validation

loss는 : [0.3100554645061493, 0.9444444179534912]

with validation

loss는 : [0.23137325048446655, 0.9166666865348816]

after preprocessing(QuantileTransformer)

loss는 : 0.17088988423347473
acc는 : 0.9166666865348816

loss는 : 0.07075485587120056
acc는 : 0.9722222089767456

'''