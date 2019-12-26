"""
ex10.py
"""
import numpy as np

from ch03.ex05 import softmax

if __name__ == '__main__':
    x = np.array([1, 2, 3])
    s = softmax(x)
    print(s)

    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    s = softmax(X)
    print(s)

    # NumPy Broadcast(브로드캐스트)
    # NumPy array의 축(axis)
    #   axis=0: row의 인덱스가 증가하는 축
    #   axis=1: column의 인덱스가 증가하는 축

    # array과 scalar 간의 브로드캐스트
    x = np.array([1, 2, 3])
    print('x =', x)
    print('x + 10 =', x + 10)

    # 2차원 array와 1차원 array 간의 브로드캐스트
    # (n, m) array와 (m,) 1차원 array는 브로드캐스트가 가능
    # (n, m) array와 (n,) 1차원 array인 경우는,
    # 1차원 array를 (n,1) shape으로 reshape를 하면 브로드캐스트가 가능.
    X = np.arange(6).reshape((2, 3))
    print('X shape:', X.shape)
    print('X =', X)

    a = np.arange(1, 4)
    print('a shape =', a.shape)
    print('a =', a)

    print('X + a =', X + a)

    b = np.array([10, 20])
    print('b shape:', b.shape)
    b = b.reshape((2, 1))
    print('b shape:', b.shape)
    print('X + b =', X + b)

    np.random.seed(1226)
    X = np.random.randint(10, size=(2, 3))
    print(X)
    # 1. X의 모든 원소들 중 최대값(m)을 찾아서,
    # X - m을 계산해서 출력
    m = X.max()
    print('X - m =', X - m)
    # 2. X의 axis=0 방향의 최대값들(각 컬럼들의 최대값)을 찾아서,
    # X의 각각의 원소에서, 그 원소가 속한 컬럼의 최대값을 뺀 행렬을 출력
    col_max = X.max(axis=0)
    print('m(axis=0)', col_max)
    print('X - col_max =', X - col_max)
    # 3. X의 axis=1 방향의 최대값들(각 row들의 최대값)을 찾아서,
    # X의 각 원소에서, 그 원소가 속한 row의 최대값을 뺀 행렬을 출력
    row_max = X.max(axis=1).reshape((2, 1))
    print('m(axis=1)', row_max)
    print('X - row_max =', X - row_max)