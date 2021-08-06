from datetime import date
import matplotlib
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

# 1. 데이터

# 1_1. 데이터 전처리

file_path = './_data/traffic_accident_data/'

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

# y_data

def year_data(xls):
    data = pd.read_excel(file_path+xls)
    day = np.array(range(4,35)).reshape(31)
    
    target = data.iloc[:,2]
    target_a = np.where(target == '사고건수')
    target_b = np.where(target == '사망자수')
    target_c = np.where(target == '부상자수')
    
    date_data = np.array(data.iloc[0:36,day])

    target_a = date_data[target_a].reshape(12*31,1)
    target_b = date_data[target_b].reshape(12*31,1)
    target_c = date_data[target_c].reshape(12*31,1)
    year_data = np.concatenate((target_a,target_b,target_c), axis=1)

    bad_num = np.where(year_data == '-')
    y_data = pd.DataFrame(year_data)
    y_data = y_data.drop(bad_num[0])

    return np.array(y_data)

y_2010 = year_data(file_t_accident_2010)
y_2011 = year_data(file_t_accident_2011)
y_2012 = year_data(file_t_accident_2012)
y_2013 = year_data(file_t_accident_2013)
y_2014 = year_data(file_t_accident_2014)

y_2015 = year_data(file_t_accident_2015)
y_2016 = year_data(file_t_accident_2016)
y_2017 = year_data(file_t_accident_2017)
y_2018 = year_data(file_t_accident_2018)
y_2019 = year_data(file_t_accident_2019)
y_2020 = year_data(file_t_accident_2020)

y_data = np.concatenate((y_2010,y_2011,y_2012,y_2013,y_2014,y_2015,
                         y_2016,y_2017,y_2018,y_2019,y_2020)
)

# print(y_data.shape) # (4018, 3)

# npy 저장

np.save('./_save/_npy/t_accident_y_data.npy', arr=y_data)