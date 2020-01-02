"""
ex02.py
f(x, y, z) = (x + y) * z
x = -2, y = 5, z = -4에서의 df/dx, df/dy, df/dz의 값을
ex01에서 구현한 MultiplyLayer와 AddLayer 클래스를 이용해서 구하세요.
    q = x + y라 하면, dq/dx = 1, dq/dy = 1
    f = q * z 이므로, df/dq = z, df/dz = q
     위의 결과를 이용하면,
    df/dx = (df/dq)(dq/dx) = z
    df/dy = (df/dq)(dq/dy) = z

numerical_gradient 함수에서 계산된 결과와 비교
"""
import numpy as np

from ch04.ex05 import numerical_gradient
from ch05.ex01_basic_layer import AddLayer, MultiplyLayer


def fn(x):
    return (x[0] + x[1]) * x[2]


def _numerical_gradient(fn, x):
    x = x.astype(np.float, copy=False)  # 실수 타입
    gradient = np.zeros_like(x) # np.zeros(shape=x.shape)
    h = 1e-4    # 0.0001
    for i in range(x.size):
        ith_value = x[i]
        x[i] = ith_value + h
        fh1 = fn(x)
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2 * h)
        x[i] = ith_value
    return gradient



if __name__ == '__main__':
    x, y, z = -2, 5, -4
    add_gate = AddLayer()
    q = add_gate.forward(x, y)
    print('q =', q)
    multi_gate = MultiplyLayer()
    f = multi_gate.forward(q, z)
    print('f =', f)

    dout = 1.0
    dq, dz = multi_gate.backward(dout)
    print('dq =', dq)
    print('dz =', dz)
    dx, dy = add_gate.backward(dq)
    print('dx =', dx)
    print('dy =', dy)

    x = np.array([-2, 5, -4])
    grad = _numerical_gradient(fn, x)
    print(grad)


    def f(x, y, z):
        return (x + y) * z

    h = 1e-12
    dx = (f(-2 + h, 5, -4) - f(-2 - h, 5, -4)) / (2 * h)
    print('df/dx =', dx)
    dy = (f(-2, 5 + h, -4) - f(-2, 5 - h, -4)) / (2 * h)
    print('df/dy =', dy)
    dz = (f(-2, 5, -4 + h) - f(-2, 5, -4 - h)) / (2 * h)
    print('df/dz =', dz)
