"""
ex09.py
2층(2-Layer) 신경망(Neural Network)
"""
import numpy as np

class TwoLayerNetwork:
    def __init__(self):
        """ 입력: 784(28x28)개
        첫번째 층(layer)의 뉴런 개수: 32개
        출력 층(layer)의 뉴런 개수: 10개
        가중치 행렬(W1, W2), bias 행렬(b1, b2)을 난수로 생성 """
        np.random.seed(1231)
        self.params = dict()    # weight/bias 행렬들을 저장하는 딕셔너리
        # x(1, 784) @ W1(784, 32) + b1(1, 32)
        self.params['W1'] = np.random.randn(784, 32)
        self.params['b1'] = np.random.randn(32)   # 1차원 행렬
        # self.b1 = np.zeros(32)   # bias가 0일 경우
        # z1(1, 32) @ W2(32, 10) + b2(1, 10)
        self.params['W2'] = np.random.randn(32, 10)
        self.params['b2'] = np.random.randn(10)


if __name__ == '__main__':
    # 신경망 생성
    neural_net = TwoLayerNetwork()
    # W1, W2, b1, b2의 shape를 확인
    print('W1 shape:', neural_net.params['W1'].shape)
    print('W2 shape:', neural_net.params['W2'].shape)
    print('b1 shape:', neural_net.params['b1'].shape)
    print('b2 shape:', neural_net.params['b2'].shape)