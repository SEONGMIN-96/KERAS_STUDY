import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    batch_size=5,
    class_mode='binary',
    shuffle=False,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './_data/brain01_data/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=False,
)
# Found 120 images belonging to 2 classes.


# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000028D21338550>
# # print(xy_train[0])
# print(xy_train[0][0])       # x값
# print(xy_train[0][1])       # y값
# # print(xy_train[0][2])     # 없음
# print(xy_train[0][0].shape, xy_train[0][1].shape)
# # (5, 150, 150, 3) (5,)

# print(xy_train[31][1])        # 마지막 배치 y
# print(xy_train[32][1])        # 없음

# print(type(xy_train))         # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>  
# print(type(xy_train[0]))      # <class 'tuple'>
# print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

# 2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(x_train, y_train)
hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32,
                    # validation_data=xy_test,
                    validation_steps=4    
) # 160/5 = 32

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할 것

print('acc :', acc[-1])
print('val_acc :', val_acc[:-1])
