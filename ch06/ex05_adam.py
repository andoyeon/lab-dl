"""
파라미터 최적화 알고리즘 4) Adam(Adaptive Moment estimate)
    AdaGrad + Momentum 알고리즘
    학습률 변화 + 속도(모멘텀) 개념 도입
    W: 파라미터
    lr: 학습률(learning rate)
    t: timestamp. 반복할 때마다 증가하는 숫자. update 메소드가 호출될 때마다 +1
    beta1, beta2: 모멘텀을 변화시킬 때 사용하는 상수들
    m: 1st momentum
    v: 2nd momentum
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W = W - lr * m_hat / sqrt(v_hat)
"""


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.99):
        self.lr = lr  # learning rate(학습률)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = dict()  # 1st momentum
        self.v = dict()  # 2nd momentum
        self.t = 0  # timestamp

    def update(self, params, gradients):
        self.t += 1








