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
from tensorflow.keras.applications import ResNet101
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))
resnet101.trainable = True

model = Sequential()
model.add(resnet101)
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(100, activation='softmax'))


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

# 결과출력
# 1. cifar 10
# trainable = True, FC : loss=1.119904637336731, acc=0.7109000086784363
# trainable = True, pool : loss=0.9314427971839905, acc=0.7411999702453613
# trainable = False, FC : loss=1.750147819519043, acc=0.553600013256073
# trainable = False, pool : loss=1.703982949256897, acc=0.5562000274658203

# 1. cifar 100
# trainable = True, FC : loss=4.31605863571167, acc=0.33169999718666077
# trainable = True, pool : loss=3.237671136856079, acc=0.3357999920845032
# trainable = False, FC : loss=5.186156272888184, acc=0.2939000129699707
# trainable = False, pool : loss=5.153144836425781, acc=0.30379998683929443
