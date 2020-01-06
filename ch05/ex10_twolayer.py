"""
ex10_twolayer.py
오차 역전파를 사용한 2층 신경망(1 은닉층 + 1 출력층)
"""
from collections import OrderedDict

import numpy as np

from ch03.ex11 import softmax
from ch05.ex05_relu import Relu
from ch05.ex07_affine import Affine
from ch05.ex08_softmax_loss import SoftmaxWithLoss
from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """ 신경망의 구조(모양) 결정 """
        np.random.seed(106)

        # 가중치/편향 행렬들을 초기화
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # layer 생성/초기화
        self.layers = OrderedDict()
        # 딕셔너리에 데이터가 추가된 순서가 유지되는 딕셔너리
        self.layers['affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu'] = Relu()
        self.layers['affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, X):
        # X = self.layers['affine1'].forward(X)
        # X = self.layers['relu'].forward(X)
        # X = self.layers['affine2'].forward(X)
        for layer in self.layers.values():
            X = layer.forward(X)
        return X

    def loss(self, X, Y_true):
        """ 입력 데이터 X와 실제 값(레이블) Y_true가 주어졌을 때,
        손실(cross-entropy)를 계산해서 리턴. """
        # 출력층(SoftmaxWithLoss) 전까지의 forward propagation을 계산
        Y_pred = self.predict(X)
        # 출력층 바로 전 계층의 출력값을 입력으로 사용해서 손실(cross-entropy) 계산
        loss = self.last_layer.forward(Y_pred, Y_true)
        return loss


if __name__ == '__main__':
    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
    # 데이터 shape 확인
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # 신경망 객체 생성
    neural_net = TwoLayerNetwork(input_size=784,
                                 hidden_size=32,    # hidden_size: 원하는 대로
                                 output_size=10)
    for key in neural_net.params:
        print(key, ':', neural_net.params[key].shape)
    for key in neural_net.layers:
        print(key, ':', neural_net.layers[key])
    print(neural_net.last_layer)

    # predict, loss 메소드 테스트
    Y_pred = neural_net.predict(X_train[0])
    print(Y_pred)
    print(np.argmax(Y_pred))

    Y_pred = neural_net.predict(X_train[:3])
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))

    loss1 = neural_net.loss(X_train[0], Y_train[0])
    print('loss1 =', loss1)

    loss2 = neural_net.loss(X_train[:3], Y_train[:3])
    print('loss2 =', loss2)