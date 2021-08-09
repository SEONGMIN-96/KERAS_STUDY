from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 1_0. scaling

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# 1_1 One_Hot_Encoding

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu',
                input_shape=(32, 32, 3)))
model.add(Conv2D(128, (2, 2), activation='relu', padding='valid'))
model.add(Conv2D(256, (2, 2), activation='relu', padding='valid'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

optimizer = Adam(lr=0.1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
es = EarlyStopping(monitor='val_loss', mode='min', patience=11)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='acc')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, verbose=1, callbacks=[es, reduce_lr],
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

걸린시간(분) :  20.521075975894927
loss :  2.0241479873657227
acc :  0.475600004196167

'''

