"""
ex02.py
"""
import numpy as np

def and_gate(x):
    # x는 [0, 0], [0, 1], [1, 0], [1, 1] 중 하나인 numpy.ndarray 타입
    # w = [w1, w2]인 numpy.ndarray 가중치와 bias b를 찾음.
    w = np.array([0.5, 0.5])    # weight
    b = 0   # bias
    y = x.dot(w) + b    # np.sum(x * w) + b
    if y >= 1:
        return 1
    else:
        return 0


def nand_gate(x):
    if and_gate(x) >= 1:
        return 0
    else:
        return 1


def or_gate(x):
    w = np.array([0.5, 0.5])
    b = 0.5
    y = x.dot(w) + b
    if y >= 1:
        return 1
    else:
        return 0





if __name__ == '__main__':
    x = np.array([1, 0])
    w = np.array([1, 1])
    test = np.sum(x * w)
    print(x)
    print(w)
    print(test)


    for i in (0, 1):
        for j in (0, 1):
            x = np.array([i, j])
            print(f'AND{x[i], x[j]}: {and_gate(x)}')

    for i in (0, 1):
        for j in (0, 1):
            x = np.array([i, j])
            print(f'NAND{x[i], x[j]}: {nand_gate(x)}')

    for i in (0, 1):
        for j in (0, 1):
            x = np.array([i, j])
            print(f'OR{x[i], x[j]}: {or_gate(x)}')







