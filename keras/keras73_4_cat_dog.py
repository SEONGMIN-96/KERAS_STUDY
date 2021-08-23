# 가장 잘나온 전이학습모델로
# 이 데이터를 학습시켜서 결과치 도출
# keras59번과의 성능 비교

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import InceptionResNetV2

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

# 전처리

inceptionResNetV2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
inceptionResNetV2.trainable = True

# 2. 모델 구성

model = Sequential()

model.add(inceptionResNetV2)
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

기존 결과
after binary_cross
val_loss : 0.8219519257545471
val_acc : 0.6450815796852112

전의 학습 결과 (InceptionResNetV2)
val_loss : 176.46202087402344
val_acc : 0.5437468886375427


'''