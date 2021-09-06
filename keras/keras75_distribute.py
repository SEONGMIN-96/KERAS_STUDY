import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try :
        tf.config.experimental.set_visible_devices([gpus[0], gpus[1]] 'GPU')

    except RuntimeError as e:
        print(e)


from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.keras import distribute
from tensorflow.python.tf2 import disable

# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
        #    tf.distribute.HierarchicalCopyAllReduce()
        #    tf.distribute.ReductionToOneDevice()    
# )

# strategy = tf.distribute.MirroredStrategy(
    # devices=['/gpu:0']
    # devices=['/gpu:1']
    # devices=['/cpu', '/gpu:0']
# )

# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    # tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    tf.distribute.experimental.CollectiveCommunication.AUTO
)

with strategy.scope():

# 2. 모델 구성

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu',
                    input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='valid'))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (2, 2), activation='relu', padding='valid'))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(128, (2, 2), activation='relu', padding='valid'))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

hist = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,
                validation_split=0.3, shuffle=True)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test, batch_size=64)

print('=========================================')
print('loss : ', loss[0])
print('acc : ', loss[1])
