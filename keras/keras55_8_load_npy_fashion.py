import numpy as np

filepath='./_save/_npy/'

x_train = np.load(filepath+'k55_x_train_fashion.npy')
y_train = np.load(filepath+'k55_y_train_fashion.npy')
x_test = np.load(filepath+'k55_x_test_fashion.npy')
y_test = np.load(filepath+'k55_y_test_fashion.npy')

from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.convolutional import Conv2D

# 1. 데이터

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape) (60000, 28, 28)
# print(x_test.shape) (10000, 28, 28)
# print(y_train.shape) (60000,)

# 1_1. One_Hot_Encoding, preprocessing

x_train = x_train.reshape(60000, 4, 196)
x_test = x_test.reshape(10000, 4, 196)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

from sklearn.preprocessing import OneHotEncoder

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LSTM, Conv1D

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(4, 196)))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, shuffle=True, callbacks=es)
end_time = time.time() - start_time
# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

# 5. plt 시각화

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1)
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2)
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

print('loss : ', loss[0])
print('acc : ', loss[1])
print('소요 시간(분) : ', end_time / 60)

plt.show()

'''



'''