"""
ex11.py
2층 신경망 테스트
"""
import numpy as np
from ch05.ex10_twolayer import TwoLayerNetwork
from dataset.mnist import load_mnist

if __name__ == '__main__':
    np.random.seed(106)

    # MNIST 데이터를 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 2층 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784,
                                 hidden_size=32,
                                 output_size=10)

    batch_size = 100    # 한번에 학습시키는 입력 데이터 개수
    learning_rate = 0.1

    iter_size = max(X_train.shape[0] // batch_size, 1)
    print(iter_size)

    for n in range(0, len(X_train), batch_size):
        X_batch = X_train[n:(n + batch_size)]
        Y_batch = Y_train[n:(n + batch_size)]

    for i in range(iter_size):
        # 처음 batch_size 개수만큼의 데이터를 입력으로 해서 gradient 계산
        grad = neural_net.gradient(X_batch, Y_batch)

        # 가중치/편향 행렬들을 수정
        for key in ('W1', 'b1', 'W2', 'b2'):
            neural_net.params[key] -= learning_rate * grad[key]

    # loss를 계산해서 출력
    loss = neural_net.loss(X_batch, Y_batch)
    print(loss)
    # accuracy를 계산해서 출력
    acc = neural_net.accuracy(X_batch, Y_batch)
    print(acc)

    epochs = 100

    # line 30 ~ 43까지의 과정을 100회(epochs)만큼 반복
    for _ in range(epochs):
        for i in range(iter_size):
            np.random.shuffle(X_batch)
            np.random.shuffle(Y_batch)
            grad = neural_net.gradient(X_batch, Y_batch)

            for key in ('W1', 'b1', 'W2', 'b2'):
                neural_net.params[key] -= learning_rate * grad[key]

        loss = neural_net.loss(X_batch, Y_batch)
        acc = neural_net.accuracy(X_batch, Y_batch)
    print(loss)
    print(acc)
    # 반복할 때마다 학습 데이터 세트를 무작위로 섞는(shuffle) 코드를 추가

    # 각 epoch마다 테스트 데이터로 테스트를 해서 accuracy를 계산

    # 100번의 epoch가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림.
    # (x = epoch, y = loss, acc)
