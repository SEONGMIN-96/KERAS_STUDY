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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. 데이터

# hh_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=0.1,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest',
# )

# hh = hh_datagen.flow_from_directory(
#     './_data/horse-or-human',
#     target_size=(200, 200),
#     batch_size=1027,
#     class_mode='binary',
#     shuffle=False,
# )
# # Found 1027 images belonging to 2 classes.

# print(hh)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000152F0658550>
# print(type(hh))
# # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(hh[0][0]))
# # <class 'numpy.ndarray'>
# print(type(hh[0]))
# # <class 'tuple'>
# print(hh[0][0].shape)        # (1027, 150, 150, 3) 
# print(hh[0][1].shape)        # (1027,)             
# print(hh[0][1])     

# # 이미지 변환이 시간이 오래걸리니, numpy파일을 저장한다.

# np.save('./_save/_npy/k59_7_hh_x.npy', arr=hh[0][0])
# np.save('./_save/_npy/k59_7_hh_y.npy', arr=hh[0][1])

# npy데이터를 로드해서 시간을 단축시킴
 
x_data = np.load('./_save/_npy/k59_7_hh_x.npy')
y_data = np.load('./_save/_npy/k59_7_hh_y.npy')

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

augment_size = (x_data.shape[0] * 20 // 100)        # 추가할 데이터의 수를 맞춰준다.

randidx = np.random.randint(x_data.shape[0], size=augment_size)
print(x_data.shape[0])      # 1027
print(randidx)              # [39310 38997 11928 ... 40079 44541 58382]
print(randidx.shape)        # (205,)

x_augmented = x_data[randidx].copy()
y_augmented = y_data[randidx].copy()

x_data = np.concatenate((x_data, x_augmented))
y_data = np.concatenate((y_data, y_augmented))

# train_test_split 활용

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        train_size=0.8, shuffle=True,
)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(200, 200, 3), padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('val_loss :', loss[0])
print('val_acc :', loss[1])

'''

val_loss : 0.5033663511276245
val_acc : 0.8106796145439148

after augment
효과 굳
val_loss : 0.3014948070049286
val_acc : 0.9068825840950012


'''