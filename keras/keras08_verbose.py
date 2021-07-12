from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

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

start = time.time()
model.fit(x, y, epochs=100, batch_size=10, verbose=1)
end = time.time() - start
print("걸린시간 :", end)

#4. 평가 예측

# loss = model.evaluate(x, y)
# print('loss 값은 : ', loss)

# result = model.predict(x_pred)
# print('result 값은 : ', result)

'''

verbose
0: 생략
1: default
2: progress bar 생략
3~: epoch only

0일때 걸린시간 : 1.3391368389129639
1일때 걸린시간 : 1.6073906421661377
2일때 걸린시간 : 1.4316036701202393
3일대 걸린시간 : 1.4008383750915527

verbose = 1 일때
batch = 1 , 10 일때 시간측정
1: 걸린시간 : 2.7350430488586426
10: 걸린시간 : 0.9218294620513916

'''