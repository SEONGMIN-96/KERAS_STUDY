import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

# to_categorical

y = to_categorical(y)

print(x.shape)      # (70000, 28, 28)
print(y.shape)      # (70000,)

# 실습
# pca를 통해 0.95 이상인게 몇개인가?

x = x.reshape(70000, 28*28)
y = y.reshape(70000, 10)

# pca = PCA(n_components=134)

# x = pca.fit_transform(x)

print(x.shape, y.shape)     # (70000, 134) (70000,)

# pcr_EVR = pca.explained_variance_ratio_
# print(pcr_EVR.shape)

# cumsum = np.cumsum(pcr_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.94)+1)

# train_test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델

model = Sequential()
model.add(Dense(64, input_shape=(28*28,)))
# model.add(Dense(64, input_shape=(134,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)

print("loss :", loss)
print("acc :", acc)


'''
기존데이터
loss : 0.17938576638698578
acc : 0.103071428835392

after pca
loss : 0.09829996526241302
acc : 0.5084999799728394

기존 784 -> 134 컬럼 축소 결과 loss는 낮아지고, acc는 증가함.
'''