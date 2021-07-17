from tensorflow.keras.datasets import cifar10
import numpy as np

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 1_1. One_Hot_Encoding

from sklearn.preprocessing import OneHotEncoder

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(30, activation='relu', kernel_size=(2, 2), input_shape=(32, 32, 3), padding='same'))
model.add(Conv2D(50, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(50, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=64, callbacks=es, verbose=1,
            validation_split=0.2, shuffle=True)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

'''

loss :  2.2132205963134766
acc :  0.5992000102996826

batch_size = 64, 모델 구성 변경

loss :  1.4203895330429077
acc :  0.6656000018119812

kernl_size=(2, 2)

loss :  1.5324794054031372
acc :  0.6518999934196472

Conv, Maxpool +1

loss :  1.1890279054641724
acc :  0.7071999907493591

'''