# 실습 cifal10 완성

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)

from sklearn.preprocessing import OneHotEncoder

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()


vgg16 = VGG16(weights='imagenet', include_top=False, 
            input_shape=(32,32,3))

vgg16.trainable=False

model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))

model.summary()

# print(len(model.weights))
# print(len(model.trainable_weights))

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)

start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25, verbose=1, callbacks=[es])
end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)

print("loss :", loss)
print("걸린 시간 :", end_time)

