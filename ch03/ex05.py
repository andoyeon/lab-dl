import numpy as np

from ch03.ex01 import sigmoid


def init_network():
    """신경망(neural network)에서 사용되는 가중치 행렬과 bias 행렬을 생성.
    교재 p.88 그림 3-20
    입력층: 입력 값 (x1, x2) -> 1x2 행렬
    은닉층:
        - 1st 은닉층: 뉴런 3개
        - 2nd 은닉층: 뉴런 2개
    출력층: 출력 값 (y1, y2) -> 1x2 행렬
    W1, W2, W3, b1, b2, b3를 난수로 생성
    """
    np.random.seed(1224)
    network = dict()  # 가중치/bias 행렬을 저장하기 위한 딕셔너리 -> 리턴 값

    # x @ W1 + b1: 1x3 행렬
    # (1x2) @ (2x3) + (1x3)
    network['W1'] = np.random.random(size=(2, 3)).round(2)
    network['b1'] = np.random.random(3).round(2)

    # z1 @ W2 + b2: 1x2 행렬
    # (1x3) @ (3x2) + (1x2)
    network['W2'] = np.random.random((3, 2)).round(2)
    network['b2'] = np.random.random(2).round(2)

    # z2 @ W3 + b3: 1x2 행렬
    # (1x2) @ (2x2) + (1x2)
    network['W3'] = np.random.random((2, 2)).round(2)
    network['b3'] = np.random.random(2).round(2)

    return network


def forward(network, x):
    """
    순방향 전파(forward propagation). 입력 -> 은닉층 -> 출력.

    :param network: 신경망에서 사용되는 가중치/bias 행렬들을 저장한 dict
    :param x: 입력 값을 가지고 있는 (1차원) 리스트. [x1, x2]
    :return: 2개의 은닉층과 출력층을 거친 후 계산된 출력 값. [y1, y2]
    """
    # 가중치 행렬:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬:
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 은닉층에서 활성화 함수: sigmoid 함수
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)  # 첫번째 은닉층 전파
    z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파
    # 출력층: z2 @ W3 + b3 값을 그대로 출력
    y = z2.dot(W3) + b3
    # return identity_function(y)  # 출력층의 활성화 함수를 적용 후 리턴
    return softmax(y)  # 출력층의 활성화 함수로 softmax 함수를 적용


# 출력층의 활성화 함수 1 - 항등 함수: 회귀(regression) 문제
def identity_function(x):
    return x


# 출력층의 활성화 함수 2 - softmax: 분류(classification) 문제
def softmax(x):
    """[x1, x2, ..., x_k, ..., x_n]일 때,
    y_k = exp(x_k) / [sum i to n exp(x_i)]
    softmnax 함수의 리턴 값은 0 ~ 1의 값이 되고, 모든 리턴 값의 총합은 1이 됨.
    이런 특징 때문에, softmax 함수의 출력값은 확률로 해석될 수 있다.
    """
    # return np.exp(x) / np.sum(np.exp(x))
    max_x = np.max(x)  # 배열 x의 원소들 중 최대값을 찾음
    y = np.exp(x - max_x) / np.sum(np.exp(x - max_x))
    return y


if __name__ == '__main__':
    # init_network() 함수 테스트
    network = init_network()
    print('W1 =', network['W1'], sep='\n')
    print('b1 =', network['b1'])
    print(network.keys())  # values(), items()

    # forward() 함수 테스트
    x = np.array([1, 2])
    y = forward(network, x)
    print('y =', y)

    # softmax() 함수 테스트
    print('x =', x)
    print('softmax =', softmax(x))

    x = [1, 2, 3]
    print('softmax =', softmax(x))

    x = [1e0, 1e1, 1e2, 1e3]  # [1, 10, 100, 1000]
    print('x =', x)
    print('softmax =', softmax(x))

