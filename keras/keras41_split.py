import numpy as np

array1 = np.array(range(1, 11))
size = 5

def split_x(a, size):
    aaa = []
    for i in range(len(a) - size + 1): 
        subset = a[i : (i + size)] 
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(array1, size)

print("dataset : \n", dataset)

x = dataset[:, :4]
y = dataset[:, 4]

print("x : \n", x)
print("y : ", y)
