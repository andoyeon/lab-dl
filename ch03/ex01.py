"""
ex01.py
Perceptron:
    - 입력: (x1, x2)
    - 출력:
        a = x1 * w1 + x2 * w2 + b 계산
        y = 1, a > 임계값
          = 0, a <= 임계값
신경망의 뉴런(neuron)에서는 입력 신호의 가중치 합을 출력값으로 변환해 주는 함수가 존재
-> 활성화 함수(activation function)
"""
import math

import numpy as np


def step_function(x):
    """
    Step Function.

    :param x: numpy.ndarray
    :return: step(계단) 함수 출력(0 또는 1)로 이루어진 numpy.ndarray
    """
    # result = []
    # for x_i in x:
    #     if x > 0:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # return np.array(result)
    y = x > 0   # [False, False, ..., True]
    return y.astype(np.int) # [0, 0, ..., 1]


def sigmoid(x):
    """ sigmoid = 1 / (1 + exp(-x)) """
    # return 1 / (1 + math.exp(-x))
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    x = np.arange(-3, 4)
    print('x =', x)
    # for x_i in x:
    #     print(step_function(x_i), end=' ')
    # print()
    print('y =', step_function(x))  # [0 0 0 0 1 1 1]
    print(sigmoid(x))
