import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, MaxPool1D, GRU, Input, concatenate

# 1. 데이터

x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
              [50,60,70],[60,70,80],[70,80,90],[80,90,100],
              [90,100,110],[100,110,120],
              [2,3,4],[3,4,5],[4,5,6]])              
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

print(x1.shape, x2.shape)   # (13, 3) (13, 3)

x1 = x1.reshape(13, 3, 1)
x2 = x2.reshape(13, 3, 1)

x1_predict = x1_predict.reshape(1, x1.shape[1], 1)
x2_predict = x2_predict.reshape(1, x1.shape[1], 1)

# 2. 모델 구성

input1 = Input(shape=(3, 1))
dense1 = SimpleRNN(units=64, activation='relu')(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
output1 = Dense(8)(dense3)

input2 = Input(shape=(3, 1))
dense11 = SimpleRNN(units=64, activation='relu')(input1)
dense12 = Dense(32, activation='relu')(dense11)
dense13 = Dense(16, activation='relu')(dense12)
output2 = Dense(8)(dense13)

merge1 = concatenate([output1, output2])
merge2 = Dense(8, name='merge2')(merge1)
merge3 = Dense(4, name='merge3', activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, epochs=100, batch_size=1, shuffle=True)

# 4. 평가, 예측

results = model.predict([x1_predict, x2_predict])

print(results)

'''

[[84.683]]

'''