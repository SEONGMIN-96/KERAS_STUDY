import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# 1. 데이터

filepath = './_save/_npy/'

x_train = np.load(filepath+'k59_3_x_train.npy')
x_test = np.load(filepath+'k59_3_x_test.npy')
y_train = np.load(filepath+'k59_3_y_train.npy')
y_test = np.load(filepath+'k59_3_y_test.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (160, 150, 150, 3) (160,) (120, 150, 150, 3) (120,)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(150, 150 ,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=40, batch_size=32, verbose=1,
         callbacks=[], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('val_loss :', loss[0])
print('val_acc :', loss[1])