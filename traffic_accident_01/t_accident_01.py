import pandas as pd
import numpy as np
import openpyxl

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
 
