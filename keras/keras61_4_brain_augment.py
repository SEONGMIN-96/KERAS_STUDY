# 훈련데이터를 기존데이터 20% 더 할것
# 완료후 기존 모델과 비교
# save_dir도 temp에 넣을 것
# 확인 후, 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain01_data/train',
    target_size=(150, 150),
    batch_size=150,
    class_mode='binary',
    shuffle=False,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './_data/brain01_data/test',
    target_size=(150, 150),
    batch_size=120,
    class_mode='binary',
    shuffle=False,
)
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(type(x_train))
print(x_train.shape)        # (150, 150, 150, 3)
print(y_train.shape)        # (150,)

# augment 를 위해서 xy_train, xy_test데이터를 배열로 분리해줌
# fit_generator 사용 안해도됨

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

augment_size = (x_train.shape[0] * 20 // 100)        # 추가할 데이터의 수를 맞춰준다.

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])      # 150
print(randidx)             
print(randidx.shape)         # (30,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape)        # (180, 150, 150, 3)
print(y_train.shape)        # (180,)

print(np.unique(y_train))

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(150, 150, 3), padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=200, batch_size=4, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('val_loss :', loss[0])
print('val_acc :', loss[1])

''''

val_loss : 3.5245423316955566
val_acc : 0.5666666626930237

'''