from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


datasets = load_boston()
x = datasets.data
y = datasets.target

x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7)

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
model.fit(x_train, y_train, epochs=100, batch_size=4)

loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict([x_test])
print('y_predict : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 : ',r2)


# 완료!!

'''

r2 :  0.7822511742500582

r2 :  0.8325312949957733

'''