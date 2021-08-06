import matplotlib.pyplot as plt
import numpy as np

x_1 = np.array([1,2,3,4])
x_2 = np.array([4,5,6,8])
y_1 = np.array([7,9,2,7])
y_2 = np.array([9,6,8,1])

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10,5))
plt.subplot(211)
plt.plot(x_1, "-", color='red')
plt.plot(y_1, "-", color='blue')
plt.legend('asdfs')
plt.subplot(212)
plt.plot(x_2, "-", color='red')
plt.plot(y_2, "-", color='blue')
plt.legend('dasf')

plt.show()
