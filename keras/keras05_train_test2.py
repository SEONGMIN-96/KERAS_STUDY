from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_train, x_test = np.split(x,[7])
y_train, y_test = np.split(y,[7])

print(x_train)
print(x_test)
print(x_train)
print(x_test)

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
