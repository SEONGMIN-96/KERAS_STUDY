# 실습 
# CNN으로 변경
# 파라미터 조정
# 노드의 개수, activation도 추가
# epochs = [1, 2, 3]
# lr 추가

#  나아아아아중 과제 : 레이어도 파라미터로 만들어본다! ex) Dense, Conv1D

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam

# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000, 28*28).astype('float32')/255
# x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델

def build_model(drop=0.5, optimizer=Adam(), lr=0.01, filters=128, activation='relu', units_1=128, units_2=128):
    inputs = Input(shape=(28, 28), name='input')
    x = Conv1D(filters=filters, kernel_size=(2), activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(units=units_1, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(units=units_2, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr), metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [100 ,200, 300]
    optimizers = [Adam]
    lr = [0.001, 0.0001]
    dropout = [0.3, 0.4, 0.5]
    filters = [128, 256, 512]
    nodes_1 = [128, 256, 512]
    nodes_2 = [128, 256, 512]
    activations = ['relu']
    epochs = [1, 2, 3]
    return {"batch_size" : batches, "optimizer" : optimizers, 
            "drop" : dropout, "filters" : filters, "activation" : activations,
            "epochs" : epochs, "units_1" : nodes_1, "units_2" : nodes_2, "lr" : lr}

hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# model2 = build_model()
model2 = KerasClassifier(build_fn=build_model, verbose=1)#, epochs=2, validation_split=0.2)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

model = RandomizedSearchCV(model2, hyperparameters, cv=5)
# model = GridSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train, y_train, verbose=1, validation_split=0.2)

# 함수형 모델에 하이퍼 파라미터 값을 넣기전에 디폴트 값을 정해주어야 모델이 구성가능하다.
# epochs의 경우 fit에 값을 입력시에 하이퍼파라미터의 epochs가 적용되지 않는다.

print(model.best_params_)      # {'activation': 'relu', 'batch_size': 1000, 'drop': 0.3, 'epochs': 3, 'filters': 512, 'optimizer': 'rmsprop'} 
print(model.best_estimator_)
print(model.best_score_)       # 0.9148666560649872   
acc = model.score(x_test, y_test)
print("최종 스코어 :", acc)     # 최종 스코어 : 0.9555000066757202 