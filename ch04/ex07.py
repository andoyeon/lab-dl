"""
경사 하강법(gradient descent)
x_new = x_init - lr * df/dx
위 과정을 반복 -> f(x)의 최솟값을 찾음
"""
import numpy as np
import matplotlib.pyplot as plt

from ch04.ex05 import numerical_gradient


def gradient_method(fn, x_init, lr=0.01, step=100):
    x = x_init  # 점진적으로 변화시킬 변수
    x_history = []  # x가 변화되는 과정을 저장할 배열
    for i in range(step):  # step 회수만큼 반복하면서
        x_history.append(x.copy())  # x의 복사본을 x 변화 과정에 기록
        grad = numerical_gradient(fn, x)  # x에서의 gradient를 계산
        x -= lr * grad  # x_new = x_init - lr * grad: x를 변경
    return x, np.array(x_history)


def fn(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


if __name__ == '__main__':
    init_x = np.array([4.])
    x, x_hist = gradient_method(fn, init_x, lr=1.5)
    print('x =', x)
    print('x_hist =', x_hist)
    # 학습률(learning rate: lr)이 너무 작으면 (lr=0.001),
    # 최솟값을 찾아가는 시간이 너무 오래 걸리고,
    # 학습률이 너무 크면(lr=1.5), 최솟값을 찾지 못하고 발산하는 경우가 생길 수 있음.

    init_x = np.array([-4., 3.])
    x, x_hist = gradient_method(fn, init_x, lr=0.1, step=100)
    print('x =', x)
    print('x_hist =', x_hist)

    # x_hist(최소값을 찾아가는 과정)을 산점도 그래프
    plt.scatter(x_hist[:, 0], x_hist[:, 1])
    # 동심원: x**2 + y**2 = r**2 -> y**2 = r**2 - x**2
    for r in range(1, 5):
        r = float(r)  # 정수 -> 실수 변환
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r**2 - x_pts**2)
        y_pts2 = -np.sqrt(r**2 - x_pts**2)
        plt.plot(x_pts, y_pts1, ':', color='gray')
        plt.plot(x_pts, y_pts2, ':', color='gray')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color='0.8')
    plt.axhline(color='0.8')
    plt.show()

