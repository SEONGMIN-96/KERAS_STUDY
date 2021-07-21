# overfit을 극복하자!!
# 1. 전체 훈련 데이터가 많이 많이!!
# 2. normalization
# 3. dropout

from tensorflow.keras.datasets import cifar100, mnist
import numpy as np
import matplotlib.pyplot as plt


# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
# (50000, 28, 28, 1) (10000, 28, 28, 1)
# print(y_train.shape, y_test.shape)

# 1_0. scaling

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

# x_trian = x_train/255.
# x_test = x_test/255.

# 1_1 One_Hot_Encoding

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


# model = Sequential()
# model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu',
#                 input_shape=(28, 28, 1)))
# model.add(Conv2D(64, (2, 2), activation='relu', padding='valid'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
# model.add(MaxPool2D())
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
# model.add(MaxPool2D())
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = load_model('./_save/keras45_1_save_model.h5')

model.summary()

# model.save('./_save/keras45_1_save_model.h5')



# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=1, callbacks=es,
                validation_split=0.1, shuffle=True)
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

걸린시간(분) :  2.4369272589683533
loss :  0.03422580659389496
acc :  0.9922999739646912

epochs = 2, load_model

걸린시간(분) :  0.30569819211959837
loss :  0.04125365614891052
acc :  0.988099992275238

epochs = 2, include

걸린시간(분) :  0.30847230354944866
loss :  0.04346940666437149
acc :  0.9866999983787537

'''


