"""
ex10_weight_compare.py
MNIST 데이터를 사용한 가중치 초기값과 신경망 성능 비교
"""
import numpy as np
import matplotlib.pyplot as plt

from ch06.ex02_sgd import Sgd
from common.multi_layer_net import MultiLayerNet
from dataset.mnist import load_mnist

# 실험 조건 세팅

weight_init_types = {
    'std=0.01': 0.01,
    'Xavier': 'sigmoid',   # 가중치 초기값: N(0, sqrt(1/n))
    'He': 'relu'   # 가중치 초기값: N(0, sqrt(2/n))
}

# 각 실험 조건별로 테스트할 신경망을 생성
neural_nets = dict()
train_losses = dict()
for key, type in weight_init_types.items():
    neural_nets[key] = MultiLayerNet(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=type)
    train_losses[key] = []  # 빈 리스트 생성 - 실험(학습)하면서 손실값들을 저장

# MNIST train/test 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

iterations = 2_000  # 학습 횟수
batch_size = 128    # 1번 학습에 사용할 샘플 개수(미니 배치)
train_size = X_train.shape[0]
optimizer = Sgd(learning_rate=0.01)   # 파라미터 최적화 알고리즘
# optimizer를 변경하면서 테스트

# 2,000번 반복하면서
for i in range(iterations):
    # 미니 배치 샘플 랜덤 추출
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]
    # 테스트 신경망 종류마다 반복
    for key, value in neural_nets.items():
        # gradient 계산
        gradients = neural_nets[key].gradient(X_batch, Y_batch)
        # 파라미터(W, b) 업데이트
        optimizer.update(neural_nets[key].params, gradients)
        # 손실(loss) 계산 -> 리스트 추가
        loss = neural_nets[key].loss(X_batch, Y_batch)
        train_losses[key].append(loss)
    # 손실 일부 출력
    if i % 300 == 0:
        print(f'===== training #{i} =====')
        for key in neural_nets:
            print(key, ':', train_losses[key][-1])

# x축-반복 횟수, y축-손실 그래프
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(iterations)
for key in neural_nets.keys():
    plt.plot(x, train_losses[key], marker=markers[key], markevery=100, label=key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0, 2.5)
plt.legend()
plt.show()
