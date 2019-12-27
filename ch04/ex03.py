"""
ex03.py
교차 엔트로피(Cross-Entropy):
    entropy = -true_value * log (expected_value)
    entropy = -sum i [t_i * log (y_i)]
"""
import pickle
import numpy as np

from ch03.ex11 import forward
from dataset.mnist import load_mnist


def cross_entropy(y_pred, y_true):
    delta = 1e-7    # log0 = -inf 가 되는 것을 방지하기 위해서 더해줄 값
    return -np.sum(y_true * np.log(y_pred + delta))



if __name__ == '__main__':
    (T_train, y_train), (T_test, y_test) = load_mnist(one_hot_label=True)

    y_true = y_test[:10]

    with open('../ch03/sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    y_pred = forward(network, T_test[:10])

    print('y_true[0] =', y_true[0]) # 숫자 7 이미지
    print('y_pred[0] =', y_pred[0]) # 7 이미지가 될 확률이 가장 큼.
    # 실제값과 예측값이 같은 경우
    print('ce =', cross_entropy(y_pred[0], y_true[0]))  # 0.0029

    print('y_true[8] =', y_true[8]) # 숫자 5 이미지
    print('y_pred[8] =', y_pred[8]) # 6 이미지가 될 확률이 가장 큼.
    # 실제값과 예측값이 다른 경우
    print('ce =', cross_entropy(y_pred[8], y_true[8]))  # 4.9094

    # 실제값과 예측값이 다를 경우 엔트로피(불확실성) 값이 커진다.
