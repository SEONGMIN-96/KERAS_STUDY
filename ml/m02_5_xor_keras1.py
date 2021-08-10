from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 1. 데이터

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [1, 0, 0, 1]

# 2. 모델

# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))    

# 3. 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100, batch_size=1)

# 4. 평가, 예측

y_predict = model.predict(x_data)
y_predict = np.round(y_predict)
print(x_data, "의 예측결과 : \n",y_predict)

results = model.evaluate(x_data, y_data)
print("model.score : ",results[1])

acc = accuracy_score(y_data, y_predict)
print("accuracy_score : ",acc)