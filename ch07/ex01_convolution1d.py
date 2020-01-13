"""
ex01_convolution.py
1차원 Convolution(합성곱), Cross-Correlation(교차상관) 연산
"""
import numpy as np


def convolution_1d(x, w):
    """ x, w: 1d ndarray, len(x) >= len(w)
    x와 w의 합성곱 연산 결과를 리턴. """
    w_r = np.flip(w)
    conv = cross_correlation_1d(x, w_r)
    return conv


def cross_correlation_1d(x, w):
    """ x, w: 1d ndarray, len(x) >= len(w)
    x와 w의 교차 상관(cross-correlation) 연산 결과를 리턴. """
    # -> convolution_1d() 함수 cross_correlation_1d()를 사용하도록 수정
    nx = len(x)  # x의 원소의 개수
    nw = len(w)  # w의 원소의 개수
    n = nx - nw + 1  # 교차 상관 연산 결과의 원소 개수
    cross_corr = []
    for i in range(n):
        x_sub = x[i:i + nw]
        fma = np.sum(x_sub * w)  # fused multiply-add
        cross_corr.append(fma)
    return np.array(cross_corr)


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print('x =', x)
    w = np.array([2, 1])
    print('w =', w)

    # Convolution(합성곱) 연산
    # 1) w를 반전
    # w_r = np.array([1, 2])
    w_r = np.flip(w)
    print('w_r =', w_r)
    # 2) FMA(Fused Multiply-Add)
    conv = []
    for i in range(4):
        x_sub = x[i:i+2]   # (0,1), (1,2), (2,3), (3,4)
        fma = np.dot(x_sub, w_r)  # np.sum(x_sub * w_r)
        conv.append(fma)
    conv = np.array(conv)
    print(conv)
    # 1차원 convolution 연산 결과의 크기(원소의 개수) = len(x) - len(w) + 1

    # convolution_1d 함수 테스트
    conv = convolution_1d(x, w)
    print(conv)

    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    conv = convolution_1d(x, w)
    print(conv)

    # 교차 상관(Cross-Correlation) 연산
    # 합성곱 연산과 다른 점은 w를 반전시키지 않는다는 것.
    # CNN(Convolution Neural Network, 합성곱 신경망)에서는
    # 가중치 행렬을 난수로 생성한 후 Gradient Descent 등을 이용해서 갱신하기 때문에,
    # 대부분의 경우 합성곱 연산 대신 교차 상관 연산을 사용함.
    cross_crr = cross_correlation_1d(x, w)
    print(cross_crr)








