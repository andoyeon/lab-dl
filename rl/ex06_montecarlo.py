"""
Monte Carlo Method(몬테 카를로 방법)
난수를 생성해서 어떤 값을 확률적으로 계산하는 방법
"""
import math
import random

n_iteration = 10000  # 전체 반복 회수
n_in = 0  # 점 (x, y)이 반지름 1인 원 안에 들어간 개수
for _ in range(n_iteration):
    # 난수 x, y를 생성해서 (x, y)
    x = random.random()
    y = random.random()
    d = math.sqrt(x**2 + y**2)  # 원점에서 점 (x, y)까지의 거리
    if d <= 1.0:
        n_in += 1
estimate_pi = (n_in / n_iteration) * 4
print(estimate_pi)











