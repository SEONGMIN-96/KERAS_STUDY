from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
# (2, 10)
 
print(x.shape)
x = np.transpose(x) # (10, 2)
print(x.shape)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # (10,)
print(y.shape)

x_pred = np.array([[10, 1.3]]) # (1, 2)
print(x_pred.shape)

#완성하시오

#2. 모델구성

model = Sequential()
model.add(Dense(4, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=3)

#4. 평가 예측

loss = model.evaluate(x, y)
print('loss 값은 : ', loss)

result = model.predict(x_pred)
print('result 값은 : ', result)

#5. 시각화

y_predict = model.predict(x)

plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.plot(y_predict, color='red')
plt.show()

'''

Epoch 2000/2000
4/4 [==============================] - 0s 667us/step - loss: 0.0012
1/1 [==============================] - 0s 73ms/step - loss: 0.0027
loss 값은 :  0.0026576011441648006
result 값은 :  [[19.885406]]

Epoch 2000/2000
4/4 [==============================] - 0s 334us/step - loss: 2.9079e-05
1/1 [==============================] - 0s 78ms/step - loss: 4.4808e-06
loss 값은 :  4.480798907025019e-06
result 값은 :  [[19.995996]]

Epoch 2000/2000
4/4 [==============================] - 0s 666us/step - loss: 2.9104e-04
1/1 [==============================] - 0s 72ms/step - loss: 2.1995e-04
loss 값은 :  0.00021995238785166293
result 값은 :  [[19.987259]]


'''



'''
행열

1) [1, 2, 3] : (3, )

2) [[1, 2, 3]] : 1행 3열

3) [[1, 2],[3, 4],[5, 6]] : 3행 2열

4) [[[1, 2, 3],[4, 5, 6]]] : 1면 2행 3열

5) [[[1, 2],[3, 4],[5, 6]]] : 1면 3행 2열

6) [[[1],[2]],[[3],[4]]] : 2면 2행 1열

'''