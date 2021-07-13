# 06_R2_2를 카피
# 함수형으로 리폼하시오.
# 서머리 확인

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

input1 = Input(shape=(1,))
dense1 = Dense(5)(input1)
dense2 = Dense(7)(dense1)
dense3 = Dense(7)(dense2)
dense4 = Dense(7)(dense3)
dense5 = Dense(4)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)
model.summary()

# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(4))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=2)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([x])
print('x_pred의 예측값 :', result)

r2 = r2_score(y, result)
print('r2 :', r2)
