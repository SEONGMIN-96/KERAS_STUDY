import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 1_1. preprocessing

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder

# y = np.reshape(y,(1,-1))
# print(y.shape)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.fit_transform(y_test).toarray()

# print(np.unique(y_train)) ## [0 1 2 3 4 5 6 7 8 9]

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D

model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(60, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                 write_graph=True, write_images=True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=100, batch_size=1024, shuffle=True, 
                 validation_split=0.2, callbacks=[es, tb]
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss는 :', loss[0])
print('acc는 :', loss[1])

'''

loss는 : 0.09243981540203094
acc는 : 0.9829999804496765

loss는 : 0.082391157746315
acc는 : 0.9830999970436096

loss는 : 0.06445126980543137
acc는 : 0.9901999831199646

loss는 : 0.07425221055746078
acc는 : 0.9914000034332275



기존의 정의

다중 분류에는 y값에 ONE_HOT_ENCODING과 categorical_crossentropy가 필요하다.

새로운 정의

O_H_E를 안하고 싶다면, sparse_categorical_entropy를 사용하면 가능.


'''