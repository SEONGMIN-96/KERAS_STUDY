# [1, np.nan, np.nan, 8, 10]

# 결측치 철
# 1. 행 삭제
# 2. 0 넣기, (특정 값) -> [1, 0, 0, 8, 10]
# 3. 앞에값               [1, 1, 1, 8, 10]
# 4. 뒤에값               [1, 8, 8, 8, 10]
# 5. 중위값               [1, 4.5, 4.5, 8, 10]
# 6. 보간
# 7. 모델링 - predict
# 8. 부스팅계열은 결측치에 대해 자유(?)롭다.

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datastrs)
print(dates)
print(type(dates))        # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print("=========================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)

'''
2021-08-13     1.000000
2021-08-14     3.333333
2021-08-15     5.666667
2021-08-16     8.000000
2021-08-17    10.000000
dtype: float64
'''

dd = Series([1, np.nan, np.nan, 4, 10]).interpolate()   # 앞뒤 수에 영향을 받음 (앞뒤가 이상치라면?)
print(dd) 
