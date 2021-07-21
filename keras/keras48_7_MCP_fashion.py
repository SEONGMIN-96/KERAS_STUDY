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

x_train = x_train.reshape(60000, 4, 196)
x_test = x_test.reshape(10000, 4, 196)

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

from sklearn.preprocessing import OneHotEncoder

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

# 2. 모델 구성

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LSTM, Conv1D

# model = Sequential()
# model.add(Conv1D(64, 2, input_shape=(4, 196)))
# model.add(LSTM(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = load_model('./_save/ModelCheckPoint/keras48_7_fashion_MCP.hdf5')

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='./_save/ModelCheckPoint/keras48_7_fashion_MCP.hdf5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, shuffle=True, callbacks=[es, cp])
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras48_7_fashion_model_save.h5')

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

# # 5. plt 시각화

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# # 1)
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# # 2)
# plt.subplot(2,1,2)
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

print('loss : ', loss[0])
print('acc : ', loss[1])
print('소요 시간(분) : ', end_time / 60)

plt.show()

'''

loss :  0.4963993728160858
acc :  0.9049000144004822

loss :  0.5005348324775696
acc :  0.9083999991416931

loss :  0.48103052377700806
acc :  0.9193999767303467

loss :  0.4954761266708374
acc :  0.91839998960495

after DNN

loss :  0.41167113184928894
acc :  0.8751000165939331
소요 시간(분) :  1.7912862658500672

after Dropout = 0.2

loss :  0.4307110011577606
acc :  0.8526999950408936
소요 시간(분) :  3.170887343088786

after LSTM

loss :  2.3028721809387207
acc :  0.10000000149011612
소요 시간(분) :  5.546421420574188

after Conv1D

loss :  0.382271945476532
acc :  0.8654999732971191
소요 시간(분) :  3.71423180103302

'''