from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=2)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([x])
print('x_pred의 예측값 :', result)

r2 = r2_score(y, result)
print('r2 :', r2)

# 과제 2
# R2를 심수안을 이겨라 !!
# deadline 일요일12시