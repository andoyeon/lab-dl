"""
ex04_conv_pad.py
"""
import numpy as np
from scipy.signal import convolve, correlate, correlate2d

from ch07.ex01_convolution1d import convolution_1d

if __name__ == '__main__':
    x = np.arange(1, 6)
    w = np.array([2, 0])
    print(convolution_1d(x, w))

    # 일반적인 convolution(x, w) 결과의 shape는 (4,)
    # convolution 연산에서 x 원소 중 1과 5는 연산에 1번만 기여.
    # 다른 원소들은 2번씩 기여.

    # x의 모든 원소가 convolution 연산에서 동일한 기여를 할 수 있도록 padding.
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print(convolution_1d(x_pad, w))

    # convolution 결과의 크기가 입력 데이터 x와 동일한 크기가 되도록 padding.
    x_pad = np.pad(x, pad_width=(1, 0), mode='constant', constant_values=0)
    print(convolution_1d(x_pad, w))

    x_pad = np.pad(x, pad_width=(0, 1), mode='constant', constant_values=0)
    print(convolution_1d(x_pad, w))

    # scipy.signal.convolve() 함수
    conv = convolve(x, w, mode='valid')
    print(conv)
    conv_full = convolve(x, w, mode='full')  # x의 모든 원소가 동일하게 연산에 기여.
    print(conv_full)
    conv_same = convolve(x, w, mode='same')  # x의 크기와 동일한 리턴.
    print(conv_same)

    # scipy.signal.correlate() 함수
    cross_corr = correlate(x, w, mode='valid')
    print(cross_corr)
    print(correlate(x, w, mode='full'))
    print(correlate(x, w, mode='same'))

    # scipy.signal.convolve2d(), scipy.signal.correlate2d()
    # (4, 4) 2d ndarray
    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    # (3, 3) 2d ndarray
    w = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])
    # x와 w의 교차 상관 연산(valid, full, same)
    print(correlate2d(x, w, mode='valid'))
    print(correlate2d(x, w, mode='full'))
    print(correlate2d(x, w, mode='same'))