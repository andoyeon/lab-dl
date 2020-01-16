"""
ex12_Pooling.py
"""
import numpy as np
import matplotlib.pyplot as plt

from common.util import im2col
from dataset.mnist import load_mnist


class Pooling:
    def __init__(self, fh, fw, stride=1, pad=0):
        self.fh = fh   # pooling 윈도우의 높이(height)
        self.fw = fw   # pooling 윈도우의 너비(width)
        self.stride = stride   # pooling 윈도우를 이동시키는 크기(보폭)
        self.pad = pad   # 패딩 크기
        # backward에서 사용하게 될 값
        self.x = None   # pooling 레이어로 forward되는 데이터
        self.arg_max = None   # 찾은 최대값의 인덱스

    def forward(self, x):
        """ x: (sample, channel, height, width) 모양의 4차원 배열 """
        self.x = x
        n, c, h, w = self.x.shape
        oh = (h - self.fh + 2 * self.pad) // self.stride + 1
        ow = (w - self.fw + 2 * self.pad) // self.stride + 1

        # 1) x --> im2col --> 2차원 변환
        col = im2col(x, self.fh, self.fw, self.stride, self.pad)

        # 2) 채널 별 최대값을 찾을 수 있는 모양으로 x를 reshape
        col = col.reshape(-1, self.fh * self.fw)

        # 3) 채널 별로 최대값을 찾음.
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # 4) 최대값(1차원 배열)을 reshape & transpose
        out = out.reshape(n, oh, ow, c)
        out = out.transpose(0, 3, 1, 2)

        # 5) pooling이 끝난 4차원 배열을 리턴
        return out


if __name__ == '__main__':
    # Pooling 클래스의 forward 메소드를 테스트
    # x를 (1, 3, 4, 4) 모양으로 무작위로(랜덤하게) 생성, 테스트
    np.random.seed(116)
    x = np.random.randint(10, size=(1, 3, 4, 4))
    print(x, 'shape:', x.shape)

    # Pooling 클래스의 인스턴스 생성
    pool = Pooling(fh=2, fw=2, stride=2, pad=0)
    out = pool.forward(x)
    print('pooled =', out)

    # MNIST 데이터를 로드
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False,
                                                      flatten=False)
    # 학습 데이터 중 5개를 batch로 forward
    x = x_train[:5]
    print('x shape:', x.shape)
    out = pool.forward(x)
    print('out shape:', out.shape)
    # 학습 데이터를 pyplot으로 그림.
    # forwarding된 결과(pooling 결과)를 pyplot으로 그림.
    for i in range(5):
        # subplot(행 개수, 열 개수, 순서)
        ax = plt.subplot(2, 5, (i + 1), xticks=[], yticks=[])
        plt.imshow(x[i].squeeze(), cmap='gray')
        ax2 = plt.subplot(2, 5, (i + 6), xticks=[], yticks=[])
        plt.imshow(out[i].squeeze(), cmap='gray')
    plt.show()
