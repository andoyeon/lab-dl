"""
ex04_sigmoid.py
sigmoid 함수: y = 1 / (1 + exp(-x))
dy/dx = y(1 - y) 증명
sigmoid 뉴런을 작성(forward, backward)
"""
import numpy as np

from ch03.ex01 import sigmoid


class Sigmoid:
    def __init__(self):
        # forward 메소드의 리턴값 y를 저장하기 위한 field
        self.y = None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)


if __name__ == '__main__':
    # Sigmoid 뉴런을 생성
    sigmoid_gate = Sigmoid()
    # x = 1일 때 sigmoid 함수의 리턴값(forward)
    y = sigmoid_gate.forward(x=0.0)
    print('y =', y) # x = 0일 때 sigmoid(0) = 0.5

    # x = 0에서의 sigmoid의 gradient(접선의 기울기)
    dx = sigmoid_gate.backward(dout=1.)
    print('dx =', dx)

    # 아주 작은 h에 대해서 [f(x + h) - f(x)]/h를 계산
    h = 1e-7
    dx2 = (sigmoid(0. + h) - sigmoid(0.)) / h
    print('dx2 =', dx2)

