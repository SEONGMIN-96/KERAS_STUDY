import pandas as pd
import numpy as np
import csv

# 1. 데이터

# 1_1. 데이터 전처리

file_path = './_data/traffic_accident_data/'

file_rain = '강수량/rain_2010_2020.csv'
file_temp = '기온/temp_2010_2020.csv'
file_wind = '바람/wind_2010_2020.csv'
file_humidity = '습도/humidity_2010_2020.csv'
file_t_accident_2010 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2010.xls'
file_t_accident_2011 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2011.xls'
file_t_accident_2012 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2012.xls'
file_t_accident_2013 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2013.xls'
file_t_accident_2014 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2014.xls'
file_t_accident_2015 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2015.xls'
file_t_accident_2016 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2016.xls'
file_t_accident_2017 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2017.xls'
file_t_accident_2018 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2018.xls'
file_t_accident_2019 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2019.xls'
file_t_accident_2020 = '사고건수_사망자수_부상자수/traffic_accident_seoul_2020.xls'

# pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 4, saw 2
# rain_data = pd.read_csv(file_path+file_rain, delimiter=';', header=None,)

# 데이터 전처리

# 강수량

f = open(file_path+file_rain, 'r', encoding='cp949')
data = list(csv.reader(f, delimiter=','))
data = data[12:len(data)]

xx = np.array(range(3652))
yy = np.array(range(3652,4018))

df = pd.DataFrame(data[1:4019])
print(df.loc[:,[2,3]])
df = df.loc[:,[3]]

#########################################

print(df.info())

'''

[4018 rows x 2 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4018 entries, 0 to 4017
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   3       4018 non-null   object
dtypes: object(1)
memory usage: 31.5+ KB
None

'''

#########################################

data = np.array(df)
print(data.shape) # (4018, 1)
print(type(data[1][0])) # <class 'str'>

# 강수량 empty = '' -> empty = '0'
# 데이터를 str -> int

data_empty = np.where(data == '')
df.loc[data_empty[0],:] = '0'
data = np.array(df)

print(type(data[0][0]))
print(data[0].shape)
print(data[19][0])
# print(data)

def to_float(a):
    aaa = []
    for i in range(len(a)):
        subset = float(a[i][0])
        aaa.append(subset)
    return np.array(aaa)
data = to_float(data)

data_rain = data.reshape(4018,1)
print(data_rain)

# 강수량 데이터 배열화 완료.

# # 기온

f = open(file_path+file_temp, 'r', encoding='cp949')
data = list(csv.reader(f, delimiter=','))
data = data[12:len(data)]

df = pd.DataFrame(data[0:4018])
print(df.loc[:,[2,3]])
df = df.loc[:,[3]]
print(df.info())
data = np.array(df)
print(data) # 
print(type(data[1][0])) # <class 'str'>

# 데이터를 str -> int

data_empty = np.where(data == '')
df.loc[data_empty[0],:] = '0'
data = np.array(df)

def to_float(a):
    aaa = []
    for i in range(len(a)):
        subset = float(a[i][0])
        aaa.append(subset)
    return np.array(aaa)
data = to_float(data)

data_temp = data.reshape(4018,1)

# 기온 데이터 배열화 완료.

# 바람

f = open(file_path+file_wind, 'r', encoding='cp949')
data = list(csv.reader(f, delimiter=','))
data = data[12:len(data)]

df = pd.DataFrame(data[4:4022])
print(df.loc[:,[2,3]])
df = df.loc[:,[3]]
print(df.info())
data = np.array(df)
print(data) # 
print(type(data[1][0])) # <class 'str'>

# 데이터를 str -> int

data_empty = np.where(data == '')
df.loc[data_empty[0],:] = '0'
data = np.array(df)

def to_float(a):
    aaa = []
    for i in range(len(a)):
        subset = float(a[i][0])
        aaa.append(subset)
    return np.array(aaa)
data = to_float(data)

data_wind = data.reshape(4018,1)

# 바람 데이터 배열화 완료.

# 습도

f = open(file_path+file_humidity, 'r', encoding='cp949')
data = list(csv.reader(f, delimiter=','))
data = data[12:len(data)]

df = pd.DataFrame(data[5:4023])
print(df.loc[:,[2,3]])
df = df.loc[:,[3]]
print(df.info())
data = np.array(df)
print(data) # 
print(type(data[1][0])) # <class 'str'>

# 데이터를 str -> float

data_empty = np.where(data == '')
df.loc[data_empty[0],:] = '0'
data = np.array(df)

def to_float(a):
    aaa = []
    for i in range(len(a)):
        subset = float(a[i][0])
        aaa.append(subset)
    return np.array(aaa)
data = to_float(data)

data_humidity = data.reshape(4018,1)

# 습도 데이터 배열화 완료.

# 기상 데이터 

print(data_rain[0],data_wind[0],data_temp[0],data_humidity[0])
print(data_rain.shape,data_wind.shape,data_temp.shape,data_humidity.shape)

x_data = np.concatenate((data_rain,data_wind,data_temp,data_humidity), axis=1)

print(x_data)
print(x_data.shape) # (4018, 4)

# npy 저장

np.save('./_save/_npy/t_accident_x_data.npy', arr=x_data)

import seaborn as sns
import matplotlib.pyplot as plt

x_data = pd.DataFrame(x_data)

# def make_corr(file):
#     corr = file.corr()
#     print(corr)

#     sns.heatmap(corr, cmap='virdis', annot=True)
#     plt.show()

# make_corr(x_data)

corr_df = x_data.corr()
corr_df = corr_df.apply(lambda x: round(x, 2))
print(corr_df)