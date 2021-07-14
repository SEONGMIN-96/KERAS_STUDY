from os import scandir
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


datasets = load_boston()
x = datasets.data
y = datasets.target

# x = (x-np.min(x)) / (np.max(x)-np.min(x))

from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

# scaler = MinMaxScaler()

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



print(x.shape)
print(y.shape)

# (506, 13)
# (506,)

#print(x_test)
#print(y_test)

#print(datasets.feature_names)
#print(datasets.DESCR)


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=13))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=8)

loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

r2 = r2_score(y_test, y_predict)

print('loss : ',loss)
print('r2 : ',r2)


# 완료!!

'''

r2 :  0.7822511742500582

r2 :  0.8325312949957733

MaxMinSclaer 후

r2 :  0.9294169883176484

loss :  6.082823753356934
r2 :  0.9263733028745493

StandardScaler 후

loss :  7.040530681610107
r2 :  0.9147812021690398

'''