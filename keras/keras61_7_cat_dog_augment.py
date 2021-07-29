# 훈련데이터를 기존데이터 20% 더 할것
# 완료후 기존 모델과 비교
# save_dir도 temp에 넣을 것
# 확인 후, 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터

cd_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

cd_train = cd_datagen.flow_from_directory(
    './_data/cat_dog01_data/training_set',
    target_size=(150, 150),
    batch_size=8005,
    class_mode='binary',
    shuffle=False,
)
# Found 8005 images belonging to 2 classes.

cd_test = cd_datagen.flow_from_directory(
    './_data/cat_dog01_data/test_set',
    target_size=(150, 150),
    batch_size=2023,
    class_mode='binary',
    shuffle=False,
)
# Found 2023 images belonging to 2 classes.

# print(cd_train)
# print(cd_test)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000152F0658550>
# print(type(cd_train))
# print(type(cd_test))
# # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(cd_train[0][0]))
# print(type(cd_test[0][0]))
# # <class 'numpy.ndarray'>
# print(type(cd_train[0]))
# print(type(cd_test[0]))
# # <class 'tuple'>
# print(cd_train[0][0].shape)        
# print(cd_train[0][1].shape)        
# print(cd_train[0][1])     
# print(cd_test[0][0].shape)         
# print(cd_test[0][1].shape)         
# print(cd_test[0][1])

# # 이미지 변환이 시간이 오래걸리니, numpy파일을 저장한다.

# np.save('./_save/_npy/k59_8_cd_x_train.npy', arr=cd_train[0][0])
# np.save('./_save/_npy/k59_8_cd_y_train.npy', arr=cd_train[0][1])
# np.save('./_save/_npy/k59_8_cd_x_test.npy', arr=cd_test[0][0])
# np.save('./_save/_npy/k59_8_cd_y_test.npy', arr=cd_test[0][1])

# npy데이터를 로드해서 시간을 단축시킴
 
x_train = np.load('./_save/_npy/k59_8_cd_x_train.npy')
y_train = np.load('./_save/_npy/k59_8_cd_y_train.npy')
x_test = np.load('./_save/_npy/k59_8_cd_x_test.npy')
y_test = np.load('./_save/_npy/k59_8_cd_y_test.npy')

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
print(x_train.shape[0])      # 8005
print(randidx)              # [39310 38997 11928 ... 40079 44541 58382]
print(randidx.shape)        # (1601,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape)
print(y_train.shape)

# 전처리

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(150, 150, 3), padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('val_loss :', loss[0])
print('val_acc :', loss[1])

'''

after categorical_cross
val_loss : 0.0
val_acc : 0.49975284934043884

after binary_cross
val_loss : 0.8219519257545471
val_acc : 0.6450815796852112

after augment
문제가 있음. 이유는 모르겠으나 loss가 과적합되었음.
val_loss : 0.0
val_acc : 0.49975284934043884

'''