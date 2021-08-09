from datetime import date
import pandas as pd
import numpy as np
import csv
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, Flatten, Dropout, Input, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score
from tensorflow.python.keras.layers.pooling import MaxPooling1D


# 1. 데이터

# pickle 
 
np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

x_data = np.load('./_save/_npy/t_accident_x_data.npy').astype(float)
y_data = np.load('./_save/_npy/t_accident_y_data.npy').astype(int)

x_df = pd.DataFrame(x_data)
y_df = pd.DataFrame(y_data)

xx_train = np.array(range(3652))
xx_test = np.array(range(3652,4018))

x_data = x_df.loc[xx_train,:]
x_data = np.array(x_data)

x_p = x_df.loc[xx_test,:]
x_p = np.array(x_p)

y_data = y_df.loc[xx_train,:]
y_p = y_df.loc[xx_test,:]

y_data_0 = np.array(y_data[0])
y_data_1 = np.array(y_data[1])
y_data_2 = np.array(y_data[2])

y_p_0 = np.array(y_p[0])
y_p_1 = np.array(y_p[1])
y_p_2 = np.array(y_p[2])

y_p = np.array(y_p)

print(x_p.shape)
print(y_p_0.shape)
print(y_p_0)
print(y_p_0[0])
print(y_p)
print(y_p.shape)
print(y_p[0])

# reshape

# x_data = x_data.reshape(4018, 4, 1)
x_data = x_data.reshape(3652, 4, 1)
x_p = x_p.reshape(366, 4, 1)

# train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                train_size=0.8, shuffle=True, random_state=66
)

y_train_0, y_test_0, y_train_1, y_test_1, y_train_2, y_test_2 = train_test_split(y_data_0, y_data_1,
                y_data_2, train_size=0.8, shuffle=True, random_state=66
)

# 2. 모델구성

input1 = Input(shape=(4, 1))
qq = LSTM(64, return_sequences=True)(input1)
qq = Conv1D(64, 2, activation='relu')(qq)
qq = Dropout(0.5)(qq)
qq = GlobalAveragePooling1D()(qq)
qq = Flatten()(qq)
middleput0 = Dense(512, activation='relu')(qq)

middleput1 = Dense(128, activation='relu')(middleput0)
qq = Dense(128, activation='relu')(middleput1)
qq = Dropout(0.5)(qq)
qq = Dense(32, activation='relu')(qq)
output0 = Dense(1, name='accident')(qq)

middleput2 = Dense(128, activation='relu')(middleput0)
qq = Dense(128, activation='relu')(middleput2)
qq = Dropout(0.5)(qq)
qq = Dense(64, activation='relu')(qq)
output1 = Dense(1, name='dead')(qq)

middleput3 = Dense(128, activation='relu')(middleput0)
qq = Dense(128, activation='relu')(middleput3)
qq = Dropout(0.5)(qq)
qq = Dense(64, activation='relu')(qq)
output2 = Dense(1, name='patient')(qq)

model = Model(inputs=input1, outputs=[output0, output1, output2])

model.summary()

# 3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', mode='min', patience=5)

model.compile(loss='mse', optimizer='adam', metrics=['acc', 'mae'])
hist = model.fit(x_train, [y_train_0, y_train_1, y_train_2], epochs=15, batch_size=32, verbose=1, validation_split=0.2,
            callbacks=[es])


# 4. 평가 예측

loss = model.evaluate(x_test, [y_test_0, y_test_1, y_test_2])
results = model.predict(x_p)

print('loss :' ,loss[0])
print('o0_val_loss : ',loss[1])
print('o1_val_loss : ',loss[2])
print('o2_val_loss : ',loss[3])
print('o0_val_acc : ',loss[4])
print('o1_val_acc : ',loss[5])
print('o2_val_acc : ',loss[6])
print('o0_val_mae : ',loss[7])
print('o1_val_mae : ',loss[8])
print('o2_val_mae : ',loss[9])
