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
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import time
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def image_generator(x_train, x_test):
    x_train = tf.image.resize(
    x_train, [80, 80], method='nearest', preserve_aspect_ratio=False,
    antialias=False, name=None
    )

    x_test = tf.image.resize(
    x_test, [80, 80], method='nearest', preserve_aspect_ratio=False,
    antialias=False, name=None
    )
    print(x_train.shape)
    return {'x_train':x_train, 'x_test':x_test}

ecd = OneHotEncoder()
y_train = ecd.fit_transform(y_train).toarray()
y_test = ecd.fit_transform(y_test).toarray()

inceptionresnetv2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(80,80,3))
inceptionresnetv2.trainable = True

def model(x_train, x_test, y_train, y_test):
    model = Sequential()
    model.add(inceptionresnetv2)
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
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

x_data = image_generator(x_train, x_test)
model(x_data['x_train'], x_data['x_test'], y_train, y_test)

'''
cifar 10
FC/trainable : False
loss :  1.8211373090744019  - accuracy :  0.37610000371932983
GAP/trainable : False
loss :  2.169480085372925  - accuracy :  0.17870000004768372
FC/trainable : True
loss :  0.5058783888816833  - accuracy :  0.8981999754905701
GAP/trainable : True
loss :  0.55596524477005  - accuracy :  0.8980000019073486
cifar100
FC/trainable : False
loss :  4.605153560638428  - accuracy :  0.009999999776482582
GAP/trainable : False
loss :  4.602907657623291  - accuracy :  0.010599999688565731
FC/trainable : True
loss :  1.9836857318878174  - accuracy :  0.6460000276565552
GAP/trainable : True
loss :  2.313573122024536  - accuracy :  0.6622999906539917
'''