from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# random.shuffle(x)
# random.shuffle(y)

# x_train, x_test = np.split(x,[70])
# y_train, y_test = np.split(y,[70])

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

# print(x_train)
print(x_test)
# print(y_train)
print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련 
# model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측 
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

#result = model.predict([12])
#print('result :', result)

# y_predict = model.predict([11])


# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

'''

Epoch 1000/1000
10/10 [==============================] - 0s 556us/step - loss: 4.5764
1/1 [==============================] - 0s 70ms/step - loss: 3.2734
loss :  3.2733566761016846
result : [[10.884739]]

'''
