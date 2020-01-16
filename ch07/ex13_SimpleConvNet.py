"""
ex13_SimpleConvNet.py
 Simple Convolutional Neural Network(CNN)
 p.228  그림 7-2
"""
from collections import OrderedDict

import numpy as np

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss


class SimpleConvNet:
    """
    1st hidden layer: Convolution -> ReLU -> Pooling
    2nd hidden layer: Affine -> ReLU (fully-connected network, 완전연결층)
    출력층: Affine -> SoftmaxWithLoss
    """
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_params={'filter_num': 30,
                              'filter_size': 5,
                              'pad': 0,
                              'stride': 1},
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01):
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_pad = conv_params['pad']
        filter_stride = conv_params['stride']
        input_size = input_dim[1]
        # oh = (h - self.fh + 2 * self.pad) // self.stride + 1
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        """ 인스턴스 초기화 - CNN 구성, 변수들 초기화
        input_dim: 입력 데이터 차원. MNIST인 경우 (1, 28, 28)
        conv_param: Convolution 레이어의 파라미터(filter, bias)를 생성하기 위해
        필요한 값들
            필터 개수(filter_num),
            필터 크기(filter_size = filter_height = filter_width),
            패딩 개수(pad),
            보폭(stride)
        hidden_size: Affine 계층에서 사용할 뉴런의 개수 - W 행렬의 크기
        output_size: 출력값의 원소의 개수. MNIST인 경우 10
        weight_init_std: 가중치(weight) 행렬을 난수로 초기화할 때 사용할 표준편차
        """
        # CNN layer(계층) 생성, 연결
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           filter_stride, filter_pad )
        self.layers['Relu1'] = Relu()
        self.layers['Pooling'] = Pooling(pool_h=2, pool_w=2, stride=1, pad=0)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

        # CNN layer에서 필요한 파라미터들
        self.params = dict
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = hidden_size
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        def predict(self):
            pass

        def loss(self):
            pass

        def accuracy(self):
            pass

        def gradient(self):
            pass


if __name__ == '__main__':
    # MNIST 데이터 로드
    # SimpleConvNet 생성
    # 학습 -> 테스트
    pass


