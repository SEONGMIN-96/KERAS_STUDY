from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([range(10)])

x = np.transpose(x) 

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
 
y = np.transpose(y)


x_pred = np.array([[9]]) 
print(x_pred.shape)

#완성하시오

#2. 모델구성

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(8))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(13))
model.add(Dense(10))
model.add(Dense(3))


#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)

#4. 평가 예측

loss = model.evaluate(x, y)
print('loss 값은 : ', loss)

result = model.predict(x_pred)
print('result 값은 : ', result)

y_preadict = model.predict(x)

plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])
plt.plot(x, y_preadict, color='red')
plt.show()

'''
Epoch 4000/4000
4/4 [==============================] - 0s 333us/step - loss: 0.0106
1/1 [==============================] - 0s 73ms/step - loss: 0.0063
loss 값은 :  0.006283408962190151
result 값은 :  [[9.971209   1.5389533  0.92540103]]

'''