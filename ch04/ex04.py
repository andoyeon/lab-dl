import numpy as np

a = np.array([1, 2, 3])
print('dim:', a.ndim)
print('shape:', a.shape)
print('size:', a.size)  # size: 원소 전체의 개수
print('len:', len(a))

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print('dim:', A.ndim)
print('shape:', A.shape)
print('size:', A.size)  # size = shape[0] * shape[1]
print('len:', len(A))
