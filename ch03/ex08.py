"""
ex08.py
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle

import numpy as np

from ch03.ex01 import sigmoid
from ch03.ex05 import softmax
from dataset.mnist import load_mnist


def init_network():
    """ 가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성 """
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어옴.
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1 ,b2, b3 shape 확인
    return network


def predict(network, x):
    """ 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴. """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)
    z2 = sigmoid(z1.dot(W2) + b2)
    y = z2.dot(W3) + b3
    return softmax(y)


def accuracy(label, pred):
    """ 테스트 데이터 레이블과 데이터 예측값을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴."""
    accuracy_cnt = 0
    for i in range(len(label)):
        p = np.argmax(pred[i])
        if p == label[i]:
            accuracy_cnt += 1
    return float(accuracy_cnt / len(label))


if __name__ == '__main__':
    network = init_network()
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True,
                                                      flatten=True,
                                                      one_hot_label=False)

    y_pred = predict(network, X_test)
    acc = accuracy(y_test, y_pred)
    print(acc)