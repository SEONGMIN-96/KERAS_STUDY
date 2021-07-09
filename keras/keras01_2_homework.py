# x = [1,2,3,4,5]
# y = [1,2,4,3,5]
# x_pred = [6]
# 완성한뒤, 출력결과 스샷

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3000, batch_size=2)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([x_pred])
print('x_pred의 예측값 :', result)


# Epoch 3000/3000
# 3/3 [==============================] - 0s 500us/step - loss: 0.4150
# 1/1 [==============================] - 0s 61ms/step - loss: 0.3800
# loss :  0.3800092339515686
# x_pred의 예측값 : [[5.695583]]