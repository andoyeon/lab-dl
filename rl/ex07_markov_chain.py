"""
Markov Chain
"""

import numpy as np

# random sampling
for _ in range(10):
    current_state = np.random.choice(range(3), p=[0.33, 0.33, 0.34])
    print(current_state, end=' ')
print()

# transition_probs: 전이 확률
# s0 -> s0, s1, s2, s3로 상태가 변할 확률
# s1 -> s0, s1, s2, s3로 상태가 변할 확률
# ...
# shape: (현재 상태의 개수, 미래 상태의 개수)
transition_probs = [
    [0.7, 0.2, 0.0, 0.1],  # s0 -> s0, s1, s2, s3
    [0.0, 0.0, 0.9, 0.1],  # s1 -> s0, s1, s2, s3
    [0.0, 1.0, 0.0, 0.0],  # s2 -> s0, s1, s2, s3
    [0.0, 0.0, 0.0, 1.0]  # s3 -> s0, s1, s2, s3
]

# 위와 같은 상태 전이 확률이 있을 때,
# 20번 실험 동안 상태는 어떤 식으로 바뀌는 지 출력
max_steps = 50  # 상태를 변화시키는 최대 횟수


def print_sequences():
    current_state = 0  # 현재 상태
    for step in range(max_steps):
        print(current_state, end=' ')
        if current_state == 3:
            # 현재 상태가 s3인 경우에는 다른 상태로 전이할 수 없으므로 loop를 종료
            break
        # random sampling - Monte Carlo 방법
        current_state = np.random.choice(range(4),
                                         p=transition_probs[current_state])
    else:
        # for loop가 break를 만나지 않고 전체를 모두 반복했을 때
        print('......', end='')
    print()  # 줄바꿈


for _ in range(20):
    # Markov Chain을 20번 실험
    print_sequences()










