import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D

# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1000 ,2000, 3000, 4000 ,5000]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.4, 0.4, 0.5]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}

hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# model2 = build_model()
model2 = KerasClassifier(build_fn=build_model, verbose=1)#, epochs=2, validation_split=0.2)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# model = RandomizedSearchCV(model2, hyperparameters, cv=5)
model = GridSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)

# TypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x0000022D8EFBC790> does not.
# 텐서 모델은 랜덤서치가 불가능하다. -> 그렇다면??
# 텐서플로우 모델을 사이킷런 모델형태로 감싸준다면 가능하다 (wrapping)
# 에포의 조절은 래핑시에도 가능하고, 래핑후에 fit에서도 가능하다. (에포를 두번 썻다면 나중 에포가 적용됨)
# 물론 validation_split 역시 가능하다.

print(model.best_params_)       # {'batch_size': 1000, 'drop': 0.4, 'optimizer': 'adam'}
print(model.best_estimator_)
print(model.best_score_)        # 0.9352833330631256 
acc = model.score(x_test, y_test)
print("최종 스코어 :", acc)     # 최종 스코어 : 0.958299994468689