import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try :
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    except RuntimeError as e:
        print(e)


from tensorflow.keras.datasets import cifar100, mnist
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)

# 1_0. scaling

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# x_trian = x_train/255.
# x_test = x_test/255.

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
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu',
                input_shape=(32, 32, 3)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='valid'))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2, 2), activation='relu', padding='valid'))
model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2, 2), activation='relu', padding='valid'))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=7)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, callbacks=es,
                validation_split=0.3, shuffle=True)
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

plt.show()

print('=========================================')
print('걸린시간(분.초) : ', end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])


'''

loss :  5.0935516357421875
acc :  0.2011999934911728

loss :  4.880283355712891
acc :  0.25099998712539673

loss :  4.985356330871582
acc :  0.2621000111103058

loss :  3.7693030834198
acc :  0.2930000126361847

loss :  4.0147705078125
acc :  0.29789999127388

# validation_split=0.045

loss :  3.8053157329559326
acc :  0.36579999327659607

loss :  2.41078782081604
acc :  0.39899998903274536

걸린시간 :  223.22210001945496
loss :  2.7047832012176514
acc :  0.4034999907016754

걸린시간(분) :  4.037497170766195
loss :  2.672872304916382
acc :  0.3977000117301941

# validation_split=0.2

걸린시간(분.초) :  2.9881329417228697
loss :  2.8046319484710693
acc :  0.36800000071525574

# validation_splot=0.3, batch_size=64

걸린시간(분.초) :  2.280129313468933
loss :  3.037069082260132
acc :  0.33169999718666077

'''