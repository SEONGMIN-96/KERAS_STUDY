from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)

# 1_1 One_Hot_Encoding

from sklearn.preprocessing import OneHotEncoder

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

print(y_train.shape, x_train.shape)

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(60, kernel_size=(3, 3), padding='same', activation='relu',
                input_shape=(32, 32, 3)))
model.add(Conv2D(100, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(100, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(90, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(90, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(80, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(80, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(70, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(70, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=15)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=1, callbacks=es,
                validation_split=0.2, shuffle=True)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss : ', loss[0])
print('acc : ', loss[1])


'''

basic

loss :  5.0935516357421875
acc :  0.2011999934911728

Conv, Maxpool + 1, Dense +, batch_size = 128

loss :  4.880283355712891
acc :  0.25099998712539673

Conv +1

loss :  4.985356330871582
acc :  0.2621000111103058

Conv +1

loss :  3.7693030834198
acc :  0.2930000126361847

batch_size = 256, Dense +1, Conv + 1

'''