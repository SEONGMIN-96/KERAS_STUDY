from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
 
print(x.shape)
x = np.transpose(x) # (10, 2)
print(x.shape)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # (10,)
print(y.shape)

x_pred = np.array([[10, 1.3, 1]]) 
print(x_pred.shape)

#완성하시오

#2. 모델구성

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

#4. 평가 예측

loss = model.evaluate(x, y)
print('loss 값은 : ', loss)

result = model.predict(x_pred)
print('result 값은 : ', result)

y_predict = model.predict(x)
plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)
plt.scatter(x[:,2],y)
plt.plot(y_predict, color='red')
plt.show()


'''

Epoch 4000/4000
4/4 [==============================] - 0s 667us/step - loss: 0.0046
1/1 [==============================] - 0s 64ms/step - loss: 1.0514e-04
loss 값은 :  0.00010513894812902436
result 값은 :  [[19.984032]]

'''