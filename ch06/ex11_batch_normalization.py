"""
ex11_batch_normalization.py
배치 정규화(Batch Normalization)
 신경망 각 층에 미니 배치(mini-batch)를 전달할 때마다 정규화(normalization)을
 실행하도록 강제하는 방법.
 -> 학습 속도 개선 - p.213 그림 6-18
 -> 파라미터(W, b)의 초기값에 크게 의존하지 않음. - p.214 그림 6-19
 -> 과적합(overfitting)을 억제.

 y = gamma * x + beta
 gamma 파라미터: 정규화된 미니 배치를 scale-up/down
 beta 파라미터: 정규화된 미니 배치를 이동(bias)
 배치 정규화를 사용할 때는 gamma와 beta를 초기값을 설정을 하고,
 학습을 시키면서 계속 갱신(업데이트)함.
"""
# p.213 그림 6-1을 그리세요.
# Batch Normalization을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교
import numpy as np
import matplotlib.pyplot as plt

from ch06.ex02_sgd import Sgd
from ch06.ex05_adam import Adam
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist


# 배치 정규화를 사용하는 신경망

bn_neural_net = MultiLayerNetExtend(input_size=784,
                                    hidden_size_list=[100, 100, 100, 100, 100],
                                    output_size=10,
                                    weight_init_std=0.01,
                                    use_batchnorm=True)
# 배치 정규화를 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size=784,
                                 hidden_size_list=[100, 100, 100, 100, 100],
                                 output_size=10,
                                 weight_init_std=0.01,
                                 use_batchnorm=False)

# 미니 배치를 20번 학습시키면서, 두 신경망에서 정확도(accuracy)를 기록
bn_train_acc_list = []
train_acc_list = []

(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

iterations = 20
batch_size = 128
optimizer = Sgd(learning_rate=0.01)

for i in range(iterations):
    mask = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[mask]
    Y_batch = Y_train[mask]

    for net in (bn_neural_net, neural_net):
        gradients = net.gradient(X_batch, Y_batch)
        optimizer.update(net.params, gradients)

    bn_train_acc = bn_neural_net.accuracy(X_batch, Y_batch)
    train_acc = neural_net.accuracy(X_batch, Y_batch)
    bn_train_acc_list.append(bn_train_acc)
    train_acc_list.append(train_acc)

    print('bn_acc', bn_train_acc)
    print('acc', train_acc)



# -> 그래프
x = np.arange(iterations)
plt.plot(x, bn_train_acc_list, label='Batch Normalization')
plt.plot(x, train_acc_list, label='Normal')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend()
plt.show()