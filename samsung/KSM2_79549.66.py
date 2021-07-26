import pandas as pd
from tensorflow.python.keras.constraints import MaxNorm
import xlwings as xw
import numpy as np

# 1. 데이터

filepath = './_data/'
fname_ss = '삼성전자 주가 20210721.csv'
fname_sk = 'SK주가 20210721.csv'

df_ss = pd.read_csv(filepath + fname_ss, encoding='cp949')

# print(df_ss)
# print(df_ss.columns) # ['일자', '시가', '고가', '저가', '종가', '종가 단순 5', '10', '20', '60', '120', '거래량','단순 5', '20.1', '60.1', '120.1', 'Unnamed: 15]'
# print(df_ss.loc[:,['일자', '시가', '고가', '저가', '종가', '거래량']])

# df_ss.drop(df_ss.index[2601:3601], inplace=True)

dropdate = df_ss[ (df_ss['일자'] < '2011/01/03')].index
df_ss.drop(dropdate, inplace=True)
df_ss.drop(df_ss.index[2601], inplace=True)

df_ss = df_ss.sort_values(by='일자', axis=0)
df_ss = df_ss.reset_index()
df_ss = df_ss.loc[:,['시가', '고가', '저가', '거래량', '종가']]


# print(df_ss['종가']) # (2601, 5)

df_sk = pd.read_csv(filepath + fname_sk, encoding='cp949')

dropdate = df_sk[ (df_sk['일자'] < '2011/01/03')].index
df_sk.drop(dropdate, inplace=True)
df_sk.drop(df_sk.index[2601], inplace=True)

df_sk = df_sk.sort_values(by='일자', axis=0)
df_sk = df_sk.reset_index()
df_sk = df_sk.loc[:,['시가', '고가', '저가', '거래량', '종가']]

x1 = df_ss
x2 = df_sk

#####################################

from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, PowerTransformer

# scaler = QuantileTransformer()
scaler = PowerTransformer()
# scaler = MinMaxScaler()  
scaler.fit_transform(x1)
scaler.transform(x2)

#####################################

# 1_1. dataframe_to_numpy.array

x1 = x1.to_numpy()
x2 = x2.to_numpy()

# print(x1.shape) # (2601, 5)

size = 5

def split_x(a, num):
    aaa = []
    for i in range(len(a) - num + 1 ): 
        subset = a[i : (i + num )] 
        aaa.append(subset)
    return np.array(aaa)

samsung = split_x(x1, size)
sk = split_x(x2, size)

x1_predict = samsung[[-1]]
x2_predict = sk[[-1]]

print(x1_predict.shape, x2_predict.shape)

samsung = np.delete(samsung,[-1,-2,-3,-4],0)
sk = np.delete(sk,[-1,-2,-3,-4],0)

print(samsung.shape) # (2593, 5, 5)
print(sk.shape) # (2593, 5, 5)

y = x1[:,0]
y = np.delete(y,[0,1,2,3,4,5,6,7],0)
print(y.shape) # (2593,)
print(x1_predict)

# 1_3. train_test_split

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(samsung, sk, y,
                        train_size=0.8, shuffle=True, random_state=44)

print(x1_train.shape) # (2074, 5, 5)
print(x1_test.shape)  # (519, 5, 5)

# #####################################

# x1_train = x1_train.reshape(2074, 5*5)
# x2_train = x2_train.reshape(2074, 5*5)

# x1_test = x1_test.reshape(519, 5*5)
# x2_test = x2_test.reshape(519, 5*5)

# x1_predict = x1_predict.reshape(1, 5*5)

# #####################################

# from sklearn.preprocessing import QuantileTransformer

# scaler = QuantileTransformer()
# scaler.fit_transform(x1_train)
# scaler.transform(x2_train)
# scaler.transform(x1_test)
# scaler.transform(x2_test)
# scaler.transform(x1_predict)

# #####################################

x1_train = x1_train.reshape(2074, 5, 5)
x2_train = x2_train.reshape(2074, 5, 5)

x1_test = x1_test.reshape(519, 5, 5)
x2_test = x2_test.reshape(519, 5, 5)

x1_predict = x1_predict.reshape(1, 5, 5)

#####################################

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM , Conv1D, Input, MaxPool1D, MaxPooling1D

# 2_1. ensemble

input1 = Input(shape=(5, 5))
qq = Conv1D(512, kernel_size=2)(input1)
qq = MaxPool1D()(qq)
qq = LSTM(64, activation='relu')(qq)
qq = Dense(64, activation='relu')(qq)
output1 = Dense(32, activation='relu')(qq)

input2 = Input(shape=(5, 5))
qq = Conv1D(512, kernel_size=2)(input2)
qq = MaxPool1D()(qq)
qq = LSTM(64, activation='relu')(qq)
qq = Dense(64, activation='relu')(qq)
output2 = Dense(32, activation='relu')(qq)

from tensorflow.keras.layers import concatenate

merge1 = concatenate([output1, output2])
qq = Dense(32, activation='relu')(merge1)
qq = Dense(16, activation='relu')(qq)
qq = Dense(16, activation='relu')(qq)
qq = Dense(8, activation='relu')(qq)
last_output = Dense(1)(qq)

# model = Model(inputs=[input1, input2], outputs=last_output)

import time

start_time = time.time()
filepath = './_save/samsung/day2/'
fname = 'samsung0723_1352_.0034-9176858.0000.hdf5'
model = load_model(filepath + fname)
end_time = time.time() - start_time

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

######################################################################

import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/samsung/day2/'
filename = '.{epoch:04d}-{loss:.4f}.hdf5'
modelpath = "".join([filepath, "samsung", date_time, "_", filename])

######################################################################

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                filepath=modelpath)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

start_time = time.time()
# hist = model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1,
#              callbacks=[es, mcp], validation_split=0.02, shuffle=True)
end_time = time.time() - start_time

# 4. 평가, 예측

loss = model.evaluate([x1_test, x2_test], y_test)
results = model.predict([x1_predict, x2_predict])

# # 5. plt 시각화

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# # 1)
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.show()

print('소요 시간 : ', end_time)
print('loss : ', loss[0])
print('예상주가 :', results)

'''

소요 시간 :  1.0035400390625
loss :  6043417.0
예상주가 : [[78816.61]]

소요 시간 :  0.0
loss :  6956279.5
예상주가 : [[79540.59]]

소요 시간 :  0.0
loss :  8846194.0
예상주가 : [[79549.66]]

소요 시간 :  0.0
loss :  7928504.5
예상주가 : [[79110.58]]

소요 시간 :  0.19871258735656738
loss :  9668546.0
예상주가 : [[79736.57]]


'''

