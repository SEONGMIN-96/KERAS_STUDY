import numpy as np

filepath='./_save/_npy/'

x_train = np.load(filepath+'k55_x_train_cifal100.npy')
y_train = np.load(filepath+'k55_y_train_cifal100.npy')
x_test = np.load(filepath+'k55_x_test_cifal100.npy')
y_test = np.load(filepath+'k55_y_test_cifal100.npy')

from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)

# 1_0. scaling

x_train = x_train.reshape(50000, 2, 16 * 3 * 32)
x_test = x_test.reshape(10000, 2, 16 * 3 * 32)

# x_trian = x_train/255.
# x_test = x_test/255.

# 1_1 One_Hot_Encoding

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D, LSTM, MaxPooling1D, MaxPool1D, Conv1D

model = Sequential()
model.add(Conv1D(32, 2, input_shape=(2, 1536)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, callbacks=es,
                validation_split=0.025, shuffle=True)
end_time = ( time.time() - start_time ) / 60

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test, batch_size=64)

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



print('걸린시간(분) : ', end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

print('=========================================')

plt.show()

'''


'''

