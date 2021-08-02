from datetime import date
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

# # 데이터 전처리

# # 강수량

# f = open(file_path+file_rain, 'r', encoding='cp949')
# data = list(csv.reader(f, delimiter=','))
# data = data[12:len(data)]

# df = pd.DataFrame(data[1:4019])
# print(df.loc[:,[2,3]])
# df = df.loc[:,[3]]
# print(df.info())
# data = np.array(df)
# print(data.shape) # (4018, 1)
# print(type(data[1][0])) # <class 'str'>

# # 강수량 empty = '' -> empty = '0'
# # 데이터를 str -> int

# data_empty = np.where(data == '')
# df.loc[data_empty[0],:] = '0'
# data = np.array(df)

# print(type(data[0][0]))
# print(data[0].shape)
# print(data[19][0])
# # data = float(data[19][0])
# # print(data)

# def to_float(a):
#     aaa = []
#     for i in range(len(a)):
#         subset = float(a[i][0])
#         aaa.append(subset)
#     return np.array(aaa)
# data = to_float(data)

# data_rain = data.reshape(4018,1)

# # 강수량 데이터 배열화 완료.

# # 기온

# f = open(file_path+file_temp, 'r', encoding='cp949')
# data = list(csv.reader(f, delimiter=','))
# data = data[12:len(data)]

# df = pd.DataFrame(data[0:4018])
# print(df.loc[:,[2,3]])
# df = df.loc[:,[3]]
# print(df.info())
# data = np.array(df)
# print(data) # 
# print(type(data[1][0])) # <class 'str'>

# # 데이터를 str -> int

# data_empty = np.where(data == '')
# df.loc[data_empty[0],:] = '0'
# data = np.array(df)

# def to_float(a):
#     aaa = []
#     for i in range(len(a)):
#         subset = float(a[i][0])
#         aaa.append(subset)
#     return np.array(aaa)
# data = to_float(data)

# data_temp = data.reshape(4018,1)

# # 기온 데이터 배열화 완료.

# # 바람

# f = open(file_path+file_wind, 'r', encoding='cp949')
# data = list(csv.reader(f, delimiter=','))
# data = data[12:len(data)]

# df = pd.DataFrame(data[4:4022])
# print(df.loc[:,[2,3]])
# df = df.loc[:,[3]]
# print(df.info())
# data = np.array(df)
# print(data) # 
# print(type(data[1][0])) # <class 'str'>

# # 데이터를 str -> int

# data_empty = np.where(data == '')
# df.loc[data_empty[0],:] = '0'
# data = np.array(df)

# def to_float(a):
#     aaa = []
#     for i in range(len(a)):
#         subset = float(a[i][0])
#         aaa.append(subset)
#     return np.array(aaa)
# data = to_float(data)

# data_wind = data.reshape(4018,1)

# # 바람 데이터 배열화 완료.

# # 습도

# f = open(file_path+file_humidity, 'r', encoding='cp949')
# data = list(csv.reader(f, delimiter=','))
# data = data[12:len(data)]

# df = pd.DataFrame(data[5:4023])
# print(df.loc[:,[2,3]])
# df = df.loc[:,[3]]
# print(df.info())
# data = np.array(df)
# print(data) # 
# print(type(data[1][0])) # <class 'str'>

# # 데이터를 str -> int

# data_empty = np.where(data == '')
# df.loc[data_empty[0],:] = '0'
# data = np.array(df)

# def to_float(a):
#     aaa = []
#     for i in range(len(a)):
#         subset = float(a[i][0])
#         aaa.append(subset)
#     return np.array(aaa)
# data = to_float(data)

# data_humidity = data.reshape(4018,1)

# # 습도 데이터 배열화 완료.

# # 기상 데이터 병합

# print(data_rain[0],data_wind[0],data_temp[0],data_humidity[0])
# print(data_rain.shape,data_wind.shape,data_temp.shape,data_humidity.shape)

# x_data = np.concatenate((data_rain,data_wind,data_temp,data_humidity), axis=1)

# print(x_data)
# print(x_data.shape)

# # x_data 완성

# y_data

# y_data_2010 

data = pd.read_excel(file_path+file_t_accident_2010)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2010 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2010.shape)    # (372, 3)
print(y_data_2010)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2010 == '-')
print(bad_num)

y_data_2010 = pd.DataFrame(y_data_2010)
y_data_2010 = y_data_2010.drop(bad_num[0])
print(y_data_2010)
print(y_data_2010.shape)

y_data_2010 = np.array(y_data_2010)

# y_data_2011 

data = pd.read_excel(file_path+file_t_accident_2011)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2011 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2011.shape)
print(y_data_2011)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2011 == '-')
print(bad_num)

y_data_2011 = pd.DataFrame(y_data_2011)
y_data_2011 = y_data_2011.drop(bad_num[0])
print(y_data_2011)
print(y_data_2011.shape)

y_data_2011 = np.array(y_data_2011)

# y_data_2012 

data = pd.read_excel(file_path+file_t_accident_2012)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2012 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2012.shape)
print(y_data_2012)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2012 == '-')
print(bad_num)

y_data_2012 = pd.DataFrame(y_data_2012)
y_data_2012 = y_data_2012.drop(bad_num[0])
print(y_data_2012)
print(y_data_2012.shape)

y_data_2012 = np.array(y_data_2012)

# y_data_2013 

data = pd.read_excel(file_path+file_t_accident_2013)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2013 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2013.shape)
print(y_data_2013)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2013 == '-')
print(bad_num)

y_data_2013 = pd.DataFrame(y_data_2013)
y_data_2013 = y_data_2013.drop(bad_num[0])
print(y_data_2013)
print(y_data_2013.shape)

y_data_2013 = np.array(y_data_2013)

# y_data_2014 

data = pd.read_excel(file_path+file_t_accident_2014)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2014 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2014.shape)
print(y_data_2014)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2014 == '-')
print(bad_num)

y_data_2014 = pd.DataFrame(y_data_2014)
y_data_2014 = y_data_2014.drop(bad_num[0])
print(y_data_2014)
print(y_data_2014.shape)

y_data_2014 = np.array(y_data_2014)

# y_data_2015

data = pd.read_excel(file_path+file_t_accident_2015)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2015 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2015.shape)
print(y_data_2015)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2015 == '-')
print(bad_num)

y_data_2015 = pd.DataFrame(y_data_2015)
y_data_2015 = y_data_2015.drop(bad_num[0])
print(y_data_2015)
print(y_data_2015.shape)

y_data_2015 = np.array(y_data_2015)

# y_data_2016

data = pd.read_excel(file_path+file_t_accident_2016)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2016 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2016.shape)
print(y_data_2016)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2016 == '-')
print(bad_num)

y_data_2016 = pd.DataFrame(y_data_2016)
y_data_2016 = y_data_2016.drop(bad_num[0])
print(y_data_2016)
print(y_data_2016.shape)

y_data_2016 = np.array(y_data_2016)

# y_data_2017

data = pd.read_excel(file_path+file_t_accident_2017)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2017 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2017.shape)
print(y_data_2017)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2017 == '-')
print(bad_num)

y_data_2017 = pd.DataFrame(y_data_2017)
y_data_2017 = y_data_2017.drop(bad_num[0])
print(y_data_2017)
print(y_data_2017.shape)

y_data_2017 = np.array(y_data_2017)

# y_data_2018

data = pd.read_excel(file_path+file_t_accident_2018)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2018 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2018.shape)
print(y_data_2018)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2018 == '-')
print(bad_num)

y_data_2018 = pd.DataFrame(y_data_2018)
y_data_2018 = y_data_2018.drop(bad_num[0])
print(y_data_2018)
print(y_data_2018.shape)

y_data_2018 = np.array(y_data_2018)

# y_data_2019

data = pd.read_excel(file_path+file_t_accident_2019)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2019 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2019.shape)
print(y_data_2019)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2019 == '-')
print(bad_num)

y_data_2019 = pd.DataFrame(y_data_2019)
y_data_2019 = y_data_2019.drop(bad_num[0])
print(y_data_2019)
print(y_data_2019.shape)

y_data_2019 = np.array(y_data_2019)

# y_data_2020

data = pd.read_excel(file_path+file_t_accident_2020)

day = np.array(range(4,35)).reshape(31)
print(day)

# '사고건수', '사망자수', '부상자수'의 인덱스값을 구한 후, 카테고리끼리 동일한 row로 배열을만든다.
# 3개의 배열이 완성되면 concatenate 를 이용해 병합, '사고건수','사망자수','부상자수' (1,3)의 배열완성

target = data.iloc[:,2]
target_a = np.where(target == '사고건수')
target_b = np.where(target == '사망자수')
target_c = np.where(target == '부상자수')

print(type(data))
print(data.iloc[:,day])

date_data = np.array(data.iloc[0:36,day])
# date_data = np.transpose(date_data)

print(date_data)
print(date_data.shape) # (31, 35)

target_a = date_data[target_a].reshape(12*31,1)
print(target_a.shape)
target_b = date_data[target_b].reshape(12*31,1)
print(target_b.shape)
target_c = date_data[target_c].reshape(12*31,1)
print(target_c.shape)

y_data_2020 = np.concatenate((target_a,target_b,target_c), axis=1)
print(y_data_2020.shape)
print(y_data_2020)

# 결치값 - 가 섞여있기때문에 (372, 3) 발생. 결치값 제거 시작

bad_num = np.where(y_data_2020 == '-')
print(bad_num)

y_data_2020 = pd.DataFrame(y_data_2020)
y_data_2020 = y_data_2020.drop(bad_num[0])
print(y_data_2020)
print(y_data_2020.shape)

y_data_2020 = np.array(y_data_2020)

# y_data 병합

y_data = np.concatenate((y_data_2010,y_data_2011,y_data_2012,y_data_2013,y_data_2014,
                        y_data_2015,y_data_2016,y_data_2017,y_data_2018,y_data_2019,y_data_2020))

print(y_data.shape) # (4018, 3)
print(type(y_data[0][0]))

# npy 저장

np.save('./_save/_npy/t_accident_y_data.npy', arr=y_data)