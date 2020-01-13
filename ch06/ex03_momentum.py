"""
ex03_momentum.py
파라미터 최적화 알고리즘 2) Momentum 알고리즘
v: 속도(velocity)
m: 모멘텀 상수(momentum constant)
lr: 학습률
W: 파라미터
v = m * v - lr * dL/dW
W = W + v = W + m * v - lr * dL/dW
"""
import matplotlib.pyplot as plt
import numpy as np

from ch06.ex01_matplot3d import fn, fn_derivative


class Momentum:
    def __init__(self, lr=0.01, m=0.9):
        self.lr = lr  # 학습률
        self.m = m  # 모멘텀 상수(속도 v에 곱해줄 상수)
        self.v = dict()  # 속도(각 파라미터 방향의 속도를 저장하기 위해서)

    def update(self, params, gradients):
        if not self.v:  # 비어있는 딕셔너리(empty dict)이면
            for key in params:
                # 파라미터(W, b 등)와 동일한 shape의 0으로 채워진 배열 생성
                self.v[key] = np.zeros_like(params[key])
        for key in params:
            # v = m * v - lr * dL/dW
            # self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
            self.v[key] *= self.m
            self.v[key] -= self.lr * gradients[key]
            # W = W + v
            params[key] += self.v[key]


if __name__ == '__main__':
    # Momentum 클래스의 인스턴스를 생성
    momentum = Momentum(lr=0.05)  # lr = 0.01, 0.1, 0.05

    # update 메소드 테스트
    params = {'x': -7., 'y': 2.}  # 파라미터 초깃값
    gradients = {'x': 0., 'y': 0.}  # gradient 초깃값
    x_history = []  # params['x']가 갱신되는 과정을 저장할 리스트
    y_history = []  # params['y']가 갱신되는 과정을 저장할 리스트
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    # contour 그래프에 파라미터의 갱신 값 그래프를 추가
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.title('Momentum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()







