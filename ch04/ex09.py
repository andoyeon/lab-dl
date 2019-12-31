"""
ex09.py
2층(2-Layer) 신경망(Neural Network)
"""
import numpy as np

from dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        """입력: input_size. (예) 784(28x28)개
        첫번째 층(layer)의 뉴런 개수:hidden_size. (예) 32개
        출력 층(layer)의 뉴런 개수:output. (예) 10개
        weight 행렬(W1, W2), bias 행렬(b1, b2)을 난수로 생성"""
        np.random.seed(1231)
        self.params = dict()  # weight/bias 행렬들을 저장하는 딕셔너리
        # x(1, 784) @ W1(784, 32) + b1(1, 32)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # z1(1, 32) @ W2(32, 10) + b2(1, 10)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """data -> 은닉층 -> 출력층-> 예측값
        sigmoid(data @ W1 + b1) = z
        softmax(z @ W2 + b2) = y
        y를 리턴
        """
        a1 = x.dot(self.params['W1']) + self.params['b1']
        z1 = self.sigmoid(a1)
        a2 = z1.dot(self.params['W2']) + self.params['b2']
        y = self.softmax(a2)
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """softmax = exp(x_k) / sum i to n [exp(x_i)]"""
        if x.ndim == 1:
            max_x = np.max(x)
            x -= max_x  # overflow를 방지하기 위해서
            result = np.exp(x) / np.sum(np.exp(x))
        else:  # ndim == 2
            x_t = x.T  # 행렬 x의 전치행렬(transpose)를 만듦.
            max_x = np.max(x_t, axis=0)
            x_t -= max_x
            result = np.exp(x_t) / np.sum(np.exp(x_t), axis=0)
            result = result.T
        return result

    def accuracy(self, x, y_ture):
        """

        :param x: 예측값을 구하고 싶은 데이터. 2차원 배열
        :param y_ture: 실제 레이블. 2차원 배열
        :return: 정확도
        """


if __name__ == '__main__':
    # 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784,
                                 hidden_size=32,
                                 output_size=10)
    # W1, W2, b1, b2의 shape를 확인
    print(f"W1: {neural_net.params['W1'].shape}, b1: {neural_net.params['b1'].shape}")
    print(f"W2: {neural_net.params['W2'].shape}, b1: {neural_net.params['b2'].shape}")

    # 신경망 클래스의 predict() 메소드 테스트
    # mnist 데이터 세트를 로드(dataset.load_mnist 사용)
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # X_train[0]를 신경망에 전파(propagate)시켜서 예측값 확인
    y_pred0 = neural_net.predict(X_train[0])
    print('y_pred0 =', y_pred0)
    print('y_true0 =', y_train[0])

    # X_train[:5]를 신경망에 전파시켜서 예측값 확인
    y_pred1 = neural_net.predict(X_train[:5])
    print('y_pred1 =', y_pred1)
    print('y_true =', y_train[:5])




