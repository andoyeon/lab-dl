"""
ex08_compare.py
파라미터 최적화 알고리즘 6개의 성능 비교 - 손실(loss), 정확도(accuracy)
"""
import matplotlib.pyplot as plt
import numpy as np

from ch05.ex10_twolayer import TwoLayerNetwork
from ch06.ex02_sgd import Sgd
from ch06.ex03_momentum import Momentum
from ch06.ex04_adagrad import AdaGrad
from ch06.ex05_adam import Adam
from ch06.ex06_rmsprop import RMSProp
from ch06.ex07_nesterov import Nesterov
from dataset.mnist import load_mnist


if __name__ == '__main__':
    # MNIST(손글씨 이미지) 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 최적화 알고리즘을 구현한 클래스의 인스턴스들을 dict에 저장
    optimizers = dict()
    optimizers['SGD'] = Sgd()
    optimizers['Momentum'] = Momentum()
    optimizers['Adagrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['RMSProp'] = RMSProp()
    optimizers['Nesterov'] = Nesterov()

    # 은닉층 1개, 출력층 1개로 이루어진 신경망을 optimizers 개수만큼 생성
    # 각 신경망에서 손실들을 저장할 dict를 생성
    neural_nets = dict()
    train_losses = dict()
    for key in optimizers:
        neural_nets[key] = TwoLayerNetwork(input_size=784,
                                           hidden_size=32,
                                           output_size=10)
        train_losses[key] = []  # loss들의 이력(history)를 저장

    # 각각의 신경망을 학습시키면서 loss를 계산/기록
    iterations = 2_000  # 총 학습 횟수
    batch_size = 128    # 한 번 학습에서 사용할 미니 배치 크기
    train_size = X_train.shape[0]
    np.random.seed(108)
    for i in range(iterations):  # 2,000번 학습 반복
        # 학습 데이터(X_train), 학습 레이블(Y_train)에서 미니 배치 크기만큼랜덤하게 데이터를 선택
        batch_mask = np.random.choice(train_size, batch_size)
        # 0 ~ 59,999 사이의 숫자들(train_size) 중에서 128(batch_size)개의 숫자를 임의로 선택
        


















