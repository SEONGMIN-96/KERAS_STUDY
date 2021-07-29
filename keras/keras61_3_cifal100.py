# 훈련데이터를 10만개로 증폭할 것!
# 완료후 기존 모델과 비교
# save_dir도 temp에 넣을 것

from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)        # (50000, 32, 32, 3)
# 5만개의 추가 데이터가 필요하기 때문에
# 기존의 데이터를 augment하여 10만개의 데이터를 만든다.

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
)

# 1. ImageDataGenerator를 정의                      // x,y가 튜플 형태로 뭉쳐있음
# 2. 파일에서 땡겨오려면 -> flow_from_directory()   // x,y가 나눠있음
# 3. 데이터에서 땡겨오려면 -> flow()

augment_size = 50000        # 추가할 데이터의 수를 맞춰준다.

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])     # 50000
print(randidx)              # [39310 38997 11928 ... 40079 44541 58382]
print(randidx.shape)        # (50000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# 5만개의 데이터중 무작위로 5만개의 데이터를 뽑기 위한 과정

x_augmented = x_augmented.reshape(x_augmented.shape[0], 32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

# img데이터는 4차원 reshape 과정

x_augmented = datagen.flow(x_augmented,
                        np.zeros(augment_size),
                        batch_size=augment_size, shuffle=False,
                        # save_to_dir='d:/temp'
).next()[0]

print(x_augmented.shape)        # (50000, 32, 32, 3)

# 4만개의 데이터가 생성되었으니, 기존의 데이터와 병합

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)     
# (100000, 32, 32, 3) (100000,) 10만개의 데이터 생성
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(np.unique(y_train)) 

# 원핫인코딩 실시

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(60, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('val_loss :', loss[0])
print('val_acc :', loss[1])

'''

val_loss : 4.289567470550537
val_acc : 0.2745000123977661

'''