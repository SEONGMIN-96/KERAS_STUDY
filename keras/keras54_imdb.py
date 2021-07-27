from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import binary_crossentropy

(x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=10000
)

# 실습 시작!! 완성하시오!!

print(x_train)
print(x_train.shape)            # (25000,)
print(np.unique(y_train))       # [0 1]

print(type(x_train))            # <class 'numpy.ndarray'>
print(type(y_train))

print("뉴스기사의 최대길이 :", max(len(i) for i in x_train))
# 뉴스기사의 최대길이 : 2494
print("뉴스기사의 평균길이 :", sum(map(len, x_train)) / len(x_train))
# 뉴스기사의 평균길이 : 238.71364

plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=200, padding='pre')
x_test = pad_sequences(x_test, maxlen=200, padding='pre')

print(x_train)
print(np.unique(x_train))

print(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(10000, 64))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', mode='min', patience=7)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
model.fit(x_train, y_train, validation_split=0.2, verbose=1, epochs=100, 
                batch_size=128, callbacks=[es]
)
end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss :', loss[0])
print('acc :', loss[1])
print('소요 시간 :', end_time)

'''

loss : 2.040719985961914
acc : 0.8457199931144714

'''