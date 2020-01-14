"""
ex07_convolution3d.py
"""
import numpy as np
from scipy.signal import correlate


def convolution3d(x, y):
    h, w = x.shape[1], x.shape[2]   # source의 height/width
    fh, fw = y.shape[1], y.shape[2] # 필터의 height/width
    oh = h - fh + 1  # 결과 행렬(output)의 height(row 개수)
    ow = w - fh + 1  # 결과 행렬(output)의 width(column 개수)
    # result = []
    result = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            x_sub = x[:, i:(i+fh), j:(j+fw)]
            # fma = np.sum(x_sub * y)
            # result.append(fma)
            result[i, j] = np.sum(x_sub * y)
    # result = np.array(result).reshape((oh, ow))
    return result


if __name__ == '__main__':
    np.random.seed(114)

    # (3, 4, 4) shape의 3차원 ndarray
    x = np.random.randint(10, size=(3, 4, 4))   # (c, h, w)
    print('x =', x)
    print(x.shape)
    # (3, 3, 3) shape의 3차원 ndarray
    w = np.random.randint(5, size=(3, 3, 3))
    print('w =', w)
    print(w.shape)
    conv1 = correlate(x, w, mode='valid')
    print('conv1 =', conv1)

    # 위와 동일한 결과를 작성
    conv2 = convolution3d(x, w)
    print('conv2 =', conv2)

    x = np.random.randint(10, size=(3, 28, 28))
    w = np.random.rand(3, 16, 16)
    conv1 = correlate(x, w, mode='valid')
    conv2 = convolution3d(x, w)
    print(conv1.shape, conv2.shape)
    print(conv1)
    print(conv2)
