from datetime import date
import matplotlib
from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 및 특수문자

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# pickle

np_load_old = np.load
np.load = lambda *a,**k:np_load_old(*a, allow_pickle=True,**k)

x_data = np.load('./_save/_npy/t_accident_x_data.npy').astype(float)
y_data = np.load('./_save/_npy/t_accident_y_data.npy').astype(int)

print(x_data.shape)
print(y_data.shape)

x_df = pd.DataFrame(x_data)
y_df = pd.DataFrame(y_data)

x_df = x_df.rename(columns={0:"강수량", 1:"기온", 2:"바람", 3:"습도"})
y_df = y_df.rename(columns={0:"사고발생량",1:"사망자수",2:"부상자수"})

ship_df = pd.concat([x_df,y_df],axis=1)

# 상관관계 분석

corr_df = ship_df.corr()
# corr_df = x_df.corr()
corr_df = corr_df.apply(lambda x: round(x, 2))
print(corr_df)

line = corr_df.unstack()
print(line)

# Series -> DataFrame
df = pd.DataFrame(line[line < 1].sort_values(ascending=False), columns=['corr'])
df.style.background_gradient(cmap='viridis')
print(df)

fig, ax = plt.subplots()
im = ax.imshow(corr_df, cmap='YlGnBu')

# Color Bar

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(corr_df.columns)))
ax.set_yticks(np.arange(len(corr_df.index)))

ax.set_xticklabels(corr_df.columns)
ax.set_yticklabels(corr_df.columns)

for x in range(len(corr_df.columns)):
    for y in range(len(corr_df.index)):
        ax.text(y, x, corr_df.iloc[y, x], ha='center', va='center', color='g')

fig.tight_layout()

plt.show()