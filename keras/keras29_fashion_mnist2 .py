from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.convolutional import Conv2D

# 1. 데이터

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape) (60000, 28, 28)
# print(x_test.shape) (10000, 28, 28)
# print(y_train.shape) (60000,)

# 1_1. One_Hot_Encoding, preprocessing

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

from sklearn.preprocessing import OneHotEncoder

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(30, activation='relu', kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(40, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(40, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, shuffle=True, callbacks=es)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

'''

loss :  0.4963993728160858
acc :  0.9049000144004822

loss :  0.5005348324775696
acc :  0.9083999991416931

loss :  0.48103052377700806
acc :  0.9193999767303467

loss :  0.4954761266708374
acc :  0.91839998960495


'''