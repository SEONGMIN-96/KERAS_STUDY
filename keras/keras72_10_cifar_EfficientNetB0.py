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
from tensorflow.keras.applications import EfficientNetB0
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

efficientNetB0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))
efficientNetB0.trainable = True

model = Sequential()
model.add(efficientNetB0)
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
FC/trainable : True
loss :  0.8146569132804871 accuracy :  0.83160001039505
GAP/trainable : True
loss :  0.9483062624931335 accuracy :  0.8371999859809875
FC/trainable : False
loss :  1.1019545793533325 accuracy :  0.6157000064849854
GAP/trainable : False
loss :  1.099859356880188 accuracy :  0.618399977684021
cifar100
FC/trainable : True
loss :  2.5120327472686768 accuracy :  0.5543000102043152
GAP/trainable : True
loss :  2.4737963676452637 accuracy :  0.5795999765396118
FC/trainable : False
loss :  2.559735059738159 accuracy :  0.3538999855518341
GAP/trainable : False
'''
