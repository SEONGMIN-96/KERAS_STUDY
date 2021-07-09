from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

#2. 모델 구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=500, batch_size=3)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 :', result)

# Epoch 500/500
# 2/2 [==============================] - 0s 1ms/step - loss: 0.4586
# 1/1 [==============================] - 0s 70ms/step - loss: 0.3800
# loss :  0.3800000846385956
# 6의 예측값 : [[5.6994157]]
