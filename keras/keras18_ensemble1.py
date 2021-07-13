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

from tensorflow.keras.models import Model
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

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=8, verbose=2)

# 4. 평가, 예측

result = model.evaluate([x1_test, x2_test], y1_test)
y_predict = model.predict([x1_test, x2_test])

print('loss는 :', result)

from sklearn.metrics import r2_score

r2 = r2_score(y1_test, y_predict)
print('r2는 :', r2)
