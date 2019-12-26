"""
ex11.py
mini-batch
"""
import numpy as np

from ch03.ex01 import sigmoid
from ch03.ex08 import init_network, accuracy
from dataset.mnist import load_mnist


def softmax(X):
    """
    1) X - 1차원: [x_1, x_2, ..., x_n]
    2) X - 2차원: [[x_11, x_12, ..., x_1n],
                   x_21, x_22, ..., x_2n],
                   ...]
    """
    dimension = X.ndim
    if dimension == 1:
        m = np.max(X)   # 1차원 배열의 최대값을 찾음.
        X = X - m   # 0 이하의 숫자로 변환 <- exp함수의 overflow를 방지하기 위해서.
        y = np.exp(X) / np.sum(np.exp(X))
    elif dimension == 2:
        # m = np.max(X, axis=1).reshape((len(X), 1))
        # # len(X): 2차원 리스트 X의 row의 개수
        # X = X - m
        # sum = np.sum(np.exp(X), axis=1).reshape((len(X), 1))
        # y = np.exp(X) / sum
        Xt = X.T    # X의 전치 행렬(transpose)
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T

    return y


def mini_batch(network, X_test, batch_size):
    batch_starts = [s for s in range(0, len(X_test), batch_size)]
    mini = [X_test[s:s+batch_size] for s in batch_starts]

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    y_pred = []
    for mini in X_test:
        a1 = mini.dot(W1) + b1
        z1 = sigmoid(a1)
        # 두번째 은닉층
        a2 = z1.dot(W2) + b2
        z2 = sigmoid(a2)
        # 출력층
        a3 = z2.dot(W3) + b3
        y = softmax(a3)

        sample_pred = np.argmax(y)
        y_pred.append(sample_pred)
    return np.array(y_pred)

if __name__ == '__main__':
    np.random.seed(2020)
    # 1차원 softmax 테스트
    a = np.random.randint(10, size=5)
    print(a)
    print(softmax(a))

    # 2차원 softmax 테스트
    A = np.random.randint(10, size=(2, 3))
    print(A)
    print(softmax(A))

    # (Train/Test)데이터 세트 로드.
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # 신경망 생성(W1, b1, ...)
    network = init_network()

    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)
    print(y_pred[:5])

    # 정확도(accuracy) 출력
    acc = accuracy(y_test, y_pred)
    print(acc)

