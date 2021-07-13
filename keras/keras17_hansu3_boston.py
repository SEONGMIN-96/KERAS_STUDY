from re import T
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8)

print(x.shape)
print(y.shape)

#print(x_test)
#print(y_test)

#print(datasets.feature_names)
#print(datasets.DESCR)

input1 = Input(shape=(13,))
dense1 = Dense(5)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(10)(dense4)
dense6 = Dense(10)(dense5)
dense7 = Dense(4)(dense6)
dense8 = Dense(4)(dense7)
dense9 = Dense(4)(dense8)
output1 = Dense(1)(dense9)

model = Model(inputs=input1, outputs=output1)

model.summary()

# model = Sequential()
# model.add(Dense(5, input_dim=13))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=1, validation_split=0.3, shuffle=True, verbose=0)

loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict([x_test])
print('y_predict : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 : ',r2)

