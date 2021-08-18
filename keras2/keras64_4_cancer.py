# 실습 
# CNN으로 변경
# 파라미터 조정
# 노드의 개수, activation도 추가
# epochs = [1, 2, 3]
# lr 추가

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv1D, Flatten
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True, random_state=66
)

print(x.shape)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)

# x_train = x_train.reshape(60000, 28*28).astype('float32')/255
# x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델

def build_model(drop=0.5, optimizer='adam', activation='relu', units=128):
    inputs = Input(shape=(4,), name='input')
    x = Dense(units=units, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(units=units, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(units=units, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1000 ,2000, 3000, 4000 ,5000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    nodes_1 = [128, 256, 512]
    activations = ['relu']
    epochs = [1, 2, 3]
    return {"batch_size" : batches, "optimizer" : optimizers, 
            "drop" : dropout, "activation" : activations,
            "epochs" : epochs, "units" : nodes_1}

hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# model2 = build_model()
model2 = KerasClassifier(build_fn=build_model, verbose=1)#, epochs=2, validation_split=0.2)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# model = RandomizedSearchCV(model2, hyperparameters, cv=5)
model = GridSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train, y_train, verbose=1, validation_split=0.2)

print(model.best_params_)     
print(model.best_estimator_)
print(model.best_score_)      
acc = model.score(x_test, y_test)
print("최종 스코어 :", acc)   

# 파라미터에 쓰이지않는 파라미터를 넣을경우 오류
