  
import numpy as np
import matplotlib.pyplot as plt

def tahn(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = tahn(x)

plt.plot(x, y)
plt.grid()
plt.show()
