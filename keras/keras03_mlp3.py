from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([range(10), range(21,31),
              range(201,211)])

x = np.transpose(x) 

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
 
y = np.transpose(y)


x_pred = np.array([[0, 21, 201]]) 
print(x_pred.shape)

#완성하시오

#2. 모델구성

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(8))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(3))


#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1500, batch_size=3)

#4. 평가 예측

loss = model.evaluate(x, y)
print('loss 값은 : ', loss)

result = model.predict(x_pred)
print('result 값은 : ', result)

y_predict = model.predict(x)
'''
plt.scatter(x[:,0], y[:,0])
plt.scatter(x[:,0], y[:,1])
plt.scatter(x[:,0], y[:,2])
plt.scatter(x[:,1], y[:,0])
plt.scatter(x[:,1], y[:,1])
plt.scatter(x[:,1], y[:,2])
plt.scatter(x[:,2], y[:,0])
plt.scatter(x[:,2], y[:,1])
plt.scatter(x[:,2], y[:,2])
'''
plt.scatter(x,y)
plt.plot(y_predict, color='red')
plt.show()


'''

Epoch 4000/4000
4/4 [==============================] - 0s 667us/step - loss: 0.1899
1/1 [==============================] - 0s 69ms/step - loss: 0.0233
loss 값은 :  0.023255642503499985
result 값은 :  [[ 1.0978458  1.2718247 10.003268 ]]

Epoch 6000/6000
4/4 [==============================] - 0s 334us/step - loss: 0.0080
1/1 [==============================] - 0s 71ms/step - loss: 0.0081
loss 값은 :  0.008087515830993652
result 값은 :  [[1.0163126 1.1287931 9.93852  ]]

'''