# 실습
# cifar10과 cifar100 으로 모델을 만들것
# Trainable=True, False
# FC로 만든것과 Avarge Pooling으로 만들것 비교

# 결과출력
# 1. cifar 10
# trainable = True, FC : loss=?, acc=?
# trainable = True, pool : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = False, pool : loss=?, acc=?

# 1. cifar 100
# trainable = True, FC : loss=?, acc=?
# trainable = True, pool : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = False, pool : loss=?, acc=?

from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.applications import MobileNetV2
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

mobileNetV2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))
mobileNetV2.trainable = True

model = Sequential()
model.add(mobileNetV2)
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)

start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25, verbose=1, callbacks=[es])
end_time = time.time() - start_time

loss, acc = model.evaluate(x_test, y_test)

print("loss :", loss)
print("acc :", acc)
print("걸린 시간 :", end_time)

'''
cifar 10
FC/trainable : False
loss :  2.133913516998291 accuracy :  0.20810000598430634
GAP/trainable : False
loss :  2.137791395187378 accuracy :  0.20679999887943268
FC/trainable : True
loss :  0.8141664266586304 accuracy :  0.8345000147819519
GAP/trainable : True
loss :  1.1066784858703613 accuracy :  0.8407999873161316
cifar100
FC/trainable : False
loss :  4.374685764312744 accuracy :  0.04529999941587448
GAP/trainable : False
loss :  4.382669448852539 accuracy :  0.04410000145435333
FC/trainable : True
loss :  2.399848461151123 accuracy :  0.5303999781608582
GAP/trainable : True
loss :  2.832974910736084 accuracy :  0.5745000243186951
'''