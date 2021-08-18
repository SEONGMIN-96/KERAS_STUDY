'''
1.
m43_SelectFromModel에 그리드서치 랜덤서치 적용
최적의 R2값과 피처임포턴스 구할 것

2.
위 스레드 값으로 SelectFromModel 돌려서
최적의 피처 갯수 구할 것

3.
위 피처 갯수로 피처 갯수를 조정한 뒤
그것을 이용해 다시 랜덤서치 그리드 서치해서
최적의 R2 구할 것

4.
1,3 비교               # 0.47 이상
'''

from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes    
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

x, y = load_diabetes(return_X_y=True)

print(x.shape, y.shape)     # (442, 10) (442,)

def model_1_GridSearch():
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                    train_size=0.8, shuffle=True, random_state=66
    )
  
    model = (XGBRegressor(n_jobs=-1))

    model.fit(x_train, y_train)
    # print("최적의 매개변수 :", model.best_estimator_)
    # print("best_score_ :", model.best_score_)   
    
    thresholds = np.sort(model.feature_importances_)

    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

    parameters = [
        {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
        "max_depth":[4,5,6]},
        {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
        "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
        {"n_estimators":[90, 110], "learning_rate":[0.6, 0.9, 1],
        "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
        "colsample_bylevel":[0.6, 0.7, 0.9] }]

    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)

        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        # print(select_x_train)

        selection_model = GridSearchCV(XGBRegressor(n_jobs=-1), parameters, cv=kfold, verbose=1)
        selection_model.fit(select_x_train, y_train)

        y_predict = selection_model.predict(select_x_test)

        score = r2_score(y_test, y_predict)

        print("Thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1],
                score*100
        ))


def model_1_RandomSearch():
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                    train_size=0.8, shuffle=True, random_state=66
    )
  
    model = (XGBRegressor(n_jobs=-1))

    model.fit(x_train, y_train)
    # print("최적의 매개변수 :", model.best_estimator_)
    # print("best_score_ :", model.best_score_)   
    
    thresholds = np.sort(model.feature_importances_)

    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

    parameters = [
        {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
        "max_depth":[4,5,6]},
        {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
        "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
        {"n_estimators":[90, 110], "learning_rate":[0.6, 0.9, 1],
        "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
        "colsample_bylevel":[0.6, 0.7, 0.9] }]

    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)

        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        # print(select_x_train)

        selection_model = RandomizedSearchCV(XGBRegressor(n_jobs=-1), parameters, cv=kfold, verbose=1)
        selection_model.fit(select_x_train, y_train)

        y_predict = selection_model.predict(select_x_test)

        score = r2_score(y_test, y_predict)

        print("Thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1],
                score*100
        ))

model_1_GridSearch()
# model_1_RandomSearch()