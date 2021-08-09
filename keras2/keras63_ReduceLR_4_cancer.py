from operator import mod
from re import T
import numpy as np
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

# 1. 데이터

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) (569, 30) (569,)

# print(y[:20])
# print(np.unique(y))

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
model.add(Dense(128, activation='relu', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=5, factor=0.1)
es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, verbose=1, shuffle=True, validation_split=0.2, callbacks=[es, reduce_lr])

# 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss는 :', loss[0])
print('accuracy는 :', loss[1])

print('============================예측===========================')
print(y_test[:5])
y_predict = model.predict(x_test)
print(y_predict[:5])


'''

default

loss는 : [64.0540771484375, 6.783908367156982]

QuantileTransformer

loss는 : [0.24994081258773804, 0.42225685715675354]

binary_crossentropy, accuarcy

loss는 : 0.17674781382083893
accuracy는 : 0.9298245906829834

loss는 : 0.14620491862297058
accuracy는 : 0.9385964870452881

'''