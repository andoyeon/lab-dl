import numpy as np

a = np.arange(10)
print('a =', a)

size = 3
for i in range(0, len(a), size):
    print(a[i:(i + size)])

print(a[9:12])

# 파이썬의 리스트에서의 append
b = [1, 2]
c = [3, 4, 5]
b.append(c)
print(b)

# numpy ndarray에서의 append
x = np.array([1, 2])
y = np.array([3, 4, 5])
x = np.append(x, y)
print(x)
