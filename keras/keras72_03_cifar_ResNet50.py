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
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
resnet50.trainable = False

model = Sequential()
model.add(resnet50)
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
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
# trainable = True, FC : loss=0.8531337976455688, acc=0.7436000108718872
# trainable = True, pool : loss=1.1947333812713623, acc= 0.7261999845504761
# trainable = False, FC : loss=1.4779624938964844, acc=0.5674999952316284
# trainable = False, pool : loss=1.4810702800750732, acc=0.5709999799728394

# 1. cifar 100
# trainable = True, FC : loss=2.8958041667938232, acc=0.40310001373291016
# trainable = True, pool : loss=2.7627315521240234, acc=0.3824999928474426
# trainable = False, FC : loss=4.565731048583984, acc=0.31769999861717224
# trainable = False, pool : loss=4.538201332092285, acc=0.32429999113082886