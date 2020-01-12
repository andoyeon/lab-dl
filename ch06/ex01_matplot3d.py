"""
ex01_matplot3d.py
"""
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # 3D 그래프를 그리기 위해서 반드시 import
import numpy as np


def fn(x, y):
    """f(x, y) = (1/20) * x**2 + y**2"""
    return x**2 / 20 + y**2


def fn_derivative(x, y):
    """편미분 df/dx, df/dy 튜플을 리턴."""
    return x / 10, 2 * y


if __name__ == '__main__':
    # np.linspace : 시작점과 끝점을 균일 간격으로 나눈 점들을 생성
    x = np.linspace(-10, 10, 1000)  # x 좌표들
    y = np.linspace(-10, 10, 1000)  # y 좌표들
    # 3차원 그래프를 그리기 위해서
    # np.meshgrid: 가로축과 세로축 점들을 나타내는 두 벡터를 인수로 받아 사각형 영역을 이루는 조합을 출력
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    fig = plt.figure()  # figure 생성
    ax = plt.axes(projection='3d')
    # projection 파라미터를 사용하려면 mpl_toolkits.mplot3d 패키지가 필요
    ax.contour3D(X, Y, Z,
                 100,  # 등고선의 개수
                 cmap='binary')  # 등고선 색상 맵(color map)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 등고선(contour) 그래프
    plt.contour(X, Y, Z, 100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()




