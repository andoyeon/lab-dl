"""
ex08.py
weight 행렬에 경사 하강법(gradient descent) 적용
"""
import numpy as np

from ch03.ex11 import softmax
from ch04.ex03 import cross_entropy
from ch04.ex05 import numerical_gradient

class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3)
        # 가중치 행렬(2x3 행렬)의 초기값들을 임의로 설정

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y

    def loss(self, x, y_true):
        """ 손실 함수(loss function) - cross entropy """
        y_pred = self.predict(x)    # 입력이 x일 때 출력 y의 예측값 계산
        ce = cross_entropy(y_pred, y_true)  # 크로스 엔트로피 계산
        return ce

    def gradient(self, x, t):
        """ x: 입력, t: 출력 실제 값(정답 레이블) """
        fn = lambda W: self.loss(x, t)
        return numerical_gradient(fn, self.W)


if __name__ == '__main__':
    # SimpleNetwork 클래스 객체를 생성
    network = SimpleNetwork()   # 생성자 호출 -> __init__() 메소드 호출
    print('W =', network.W)

    # x = [0.6, 0.9]일 때 y_true = [0, 0, 1]라고 가정
    x = np.array([0.6, 0.9])
    y_true = np.array([0.0, 0.0, 1.0])
    print('x =', x)
    print('y_true =', y_true)

    y_pred = network.predict(x)
    print('y_pred =', y_pred)

    ce = network.loss(x, y_true)
    print('cross entropy =', ce)

    g1 = network.gradient(x, y_true)
    print('g1 =', g1)

    lr = 0.1    # learning rate
    for _ in range(100):
        network.W -= lr * g1    # W = W - lr * gradient
        print('W =', network.W)
        print('y_pred =', network.predict(x))
        print('ce =', network.loss(x, y_true))