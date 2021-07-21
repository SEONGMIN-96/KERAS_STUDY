#  실습
#  1~100까지의 데이터를

# 1. 데이터

from enum import Flag
import numpy as np

x_data = np.array(range(1, 101))
x_data_p = np.array(range(96, 105))

size = 5
size2 = 4

def split_x(a, num):
    aaa = []
    for i in range(len(a) - num + 1 ): 
        subset = a[i : (i + num )] 
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)
dataset_p = split_x(x_data_p, size2)

# print("dataset : \n", dataset)

x = dataset[:, :4]
y = dataset[:, 4]

x_predict = dataset_p[:, :4]

print("dataset_p \n", dataset_p)
print("x_predict : \n", x_predict)
# print("x : \n", x)
# print("y : ", y)

# 1_1. train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                            train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, x_test.shape, x_predict.shape)  # (76, 4) (20, 4) (6, 4)

x_train = x_train.reshape(76, 4, 1)
x_test = x_test.reshape(20, 4, 1)

x_predict = x_predict.reshape(6, 4, 1)

# # 1_2. One_Hot_Encoding

# from sklearn.preprocessing import OneHotEncoder

# ecd = OneHotEncoder()
# y_train = ecd.fit_transform(y_train)
# y_test = ecd.fit_transform(y_test)

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Flatten

model = Sequential()
# model.add(LSTM(64, input_shape=(4, 1), activation='relu'))
model.add(Conv1D(64, 2, input_shape=(4, 1)))
# model.add(Flatten())
model.add(LSTM())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, callbacks=es, validation_split=0.2)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
results = model.predict(x_predict)

from sklearn.metrics import r2_score

y_predict = np.array([101, 102, 103, 104, 105, 106])

r2 = r2_score(y_predict, results)

print('results : \n', results)
print('loss : ', loss[0])
print('mae : ', loss[1])
print('r2 : ', r2)


'''

after LSTM

results :  
[[100.51397]
 [101.54169]
 [102.56947]
 [103.59729]
 [104.62517]
 [105.6531 ]]
loss :  0.34114155173301697
mae :  0.5010517239570618
r2 :  0.9397345997326608


'''