import pandas as pd
import numpy as np
import csv

# x_data

file_path = './_data/traffic_accident_data/'

file_rain = '강수량/rain_2010_2020.csv'
file_temp = '기온/temp_2010_2020.csv'
file_wind = '바람/wind_2010_2020.csv'
file_humidity = '습도/humidity_2010_2020.csv'

def to_float(a):
    aaa = []
    for i in range(len(a)):
        subset = float(a[i][0])
        aaa.append(subset)
    return np.array(aaa)

def to_np(fname, size0, size1):
    f = open(file_path+fname, 'r', encoding='cp949')
    data = list(csv.reader(f, delimiter=','))
    data = data[12:len(data)]

    df = pd.DataFrame(data[size0:size1])
    df = df.loc[:,[3]]
    data = np.array(df)

    data_empty = np.where(data == '')
    df.loc[data_empty[0],:] = '0'
    data = np.array(df)
    data = to_float(data)
    
    return data.reshape(4018,1)

rain_data = to_np(file_rain, 1, 4019)
temp_data = to_np(file_temp, 0, 4018)
wind_data = to_np(file_wind, 4, 4022)
humidity_data = to_np(file_humidity, 5, 4023)

x_data = np.concatenate((rain_data, temp_data, wind_data, humidity_data), axis=1)

# npy 저장

np.save('./_save/_npy/t_accident_x_data.npy', arr=x_data)
