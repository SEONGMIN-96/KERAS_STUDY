from re import M
import numpy as np

# 1. 데이터

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array(range(1001, 1101))

# print(np.shape(x1), np.shape(x2), np.shape(y1)) # (100, 3) (100, 3) (100,)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2,
                        train_size=0.7, shuffle=False)
y1_train, y1_test = train_test_split(y1,
                        train_size=0.7, shuffle=False)

# 2. 모델 구성

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input

# 2-1. 모델1

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(4)(dense3)

# 2-2. 모델2

input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(7, activation='relu', name='dense12')(dense11)
dense13 = Dense(5, activation='relu', name='dense13')(dense12)
dense14 = Dense(5, activation='relu', name='dense14')(dense13)
output2 = Dense(4)(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)

last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
        restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
        filepath='./_save/ModelCheckPoint/keras49_mcp.h5')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2,
        callbacks=[es, mcp])

model.save('./_save/ModelCheckPoint/keras49_model_save.h5')

from sklearn.metrics import r2_score

print("======================기본출력=======================")

# 4. 평가, 예측

result = model.evaluate([x1_test, x2_test], y1_test)
y_predict = model.predict([x1_test, x2_test])

print('model의 loss :', result)

r2 = r2_score(y1_test, y_predict)
print('model의 r2 :', r2)

print("======================1. load_model =======================")

model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.h5')

result = model2.evaluate([x1_test, x2_test], y1_test)
y_predict = model2.predict([x1_test, x2_test])

print('model2의 loss :', result)

r2 = r2_score(y1_test, y_predict)
print('model2의 r2 :', r2)

print("======================2. Model_Check_Point =======================")

model3 = load_model('./_save/ModelCheckPoint/keras49_mcp.h5')

result = model3.evaluate([x1_test, x2_test], y1_test)
y_predict = model3.predict([x1_test, x2_test])

print('model3의 loss :', result)

r2 = r2_score(y1_test, y_predict)
print('model3의 r2 :', r2)

'''

======================기본출력=======================
1/1 [==============================] - 0s 15ms/step - loss: 15173.0469 - mae: 121.8719
model의 loss : [15173.046875, 121.87185668945312]
model의 r2 : -201.53232114706873
======================1. load_model =======================
1/1 [==============================] - 0s 92ms/step - loss: 15173.0469 - mae: 121.8719
model2의 loss : [15173.046875, 121.87185668945312]
model2의 r2 : -201.53232114706873
======================2. Model_Check_Point =======================
1/1 [==============================] - 0s 101ms/step - loss: 11682.0889 - mae: 106.6032
model3의 loss : [11682.0888671875, 106.60320281982422]
model3의 r2 : -154.9344411390443

earlystopping의 restore_best_weights=True = ModelCheckPoint의 save_best_only=True
와 동일

======================기본출력=======================
1/1 [==============================] - 0s 15ms/step - loss: 274.6945 - mae: 13.8220
model의 loss : [274.6944580078125, 13.822030067443848]
model의 r2 : -2.66666701891531
======================1. load_model =======================
1/1 [==============================] - 0s 91ms/step - loss: 274.6945 - mae: 13.8220
model2의 loss : [274.6944580078125, 13.822030067443848]
model2의 r2 : -2.66666701891531
======================2. Model_Check_Point =======================
1/1 [==============================] - 0s 85ms/step - loss: 274.6945 - mae: 13.8220
model3의 loss : [274.6944580078125, 13.822030067443848]
model3의 r2 : -2.66666701891531

'''