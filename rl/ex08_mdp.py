"""
MDP(Markov Decision Process)
1. Q-Value Iteration Algorithm(가치 반복 알고리즘)
2. Q-Learning (Q-학습)
"""
import numpy as np
import matplotlib.pyplot as plt

# 상태 공간(state-space): [s0, s1, s2]
# 행동 공간(action-space): [a0, a1, a2]

# Q-Value Iteration
transition_probs = [  # shape: (s, a, s')
    # 현재 상태가 s0일 때,
    [  # shape: (a, s')
        # s0에서 a0 행동을 했을 때, s0, s1, s2로 전이될 확률
        [0.7, 0.3, 0.0],
        # s0에서 a1 행동을 했을 때, s0, s1, s2로 전이될 확률
        [1.0, 0.0, 0.0],
        # s0에서 a2 행동을 했을 때, s0, s1, s2로 전이될 확률
        [0.8, 0.2, 0.0]
    ],
    # 현재 상태가 s1일 때,
    [
        [0.0, 1.0, 0.0],  # s1에서 a0 행동을 할 때
        None,  # s1에서 a1 행동을 할 때
        [0.0, 0.0, 1.0]  # s1에서 a2 행동을 할때
    ],
    # 현재 상태가 s2일 때,
    [
        None,  # (s2, a0) -> s0, s1, s2
        [0.8, 0.1, 0.1],  # (s2, a1) -> s0, s1, s2
        None,  # (s2, a2) -> s0, s1, s2
    ]
]

rewards = [  # shape: (s, a, s')
    [
        [10, 0, 0],  # (s0, a0) -> s0, s1, s2
        [0, 0, 0],  # (s0, a1) -> s0, s1, s2
        [0, 0, 0]  # (s0, a2) -> s0, s1, s2
    ],
    [
        [0, 0, 0],  # (s1, a0) -> s0, s1, s2
        [0, 0, 0],  # (s1, a1) -> s0, s1, s2
        [0, 0, -50]  # (s1, a2) -> s0, s1, s2
    ],
    [
        [0, 0, 0],  # (s2, a0) -> s0, s1, s2
        [40, 0, 0],  # (s2, a1) -> s0, s1, s2
        [0, 0, 0]  # (s2, a2) -> s0, s1, s2
    ]
]

# 각 상태(s)에서 가능한 action(a)들의 리스트
possible_actions = [
    [0, 1, 2], [0, 2], [1]
]

# Q(s, a)
Q_values = np.full(shape=(3, 3), fill_value=-np.inf)
print(Q_values)
# 모든 상태(s)와 행동(a)에 대해서 Q_value의 값들을 0으로 초기화
for state, action in enumerate(possible_actions):
    Q_values[state, action] = 0.0
print(Q_values)

gamma = 0.95  # 할인율
history_q_iter = []  # Q-value 반복 알고리즘 추정값들을 저장할 리스트

for iteration in range(50):
    # Q0 -> Q1 -> Q2 -> ... -> Q_k -> Q_k+1 -> ...-> Q49
    Q_prev = Q_values.copy()  # Q_values 리스트가 계속 갱신되기 때문에
    history_q_iter.append(Q_prev)  # 복사한 값을 history에 추가
    for s in range(3):  # 가능한 상태(state) 개수만큼 반복
        for a in possible_actions[s]:  # 그 상태(state)에서 가능한 행동의 개수만큼
            Q_values[s, a] = np.sum(
                [transition_probs[s][a][sp] *
                 (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                 for sp in range(3)]
            )

print(Q_values)
print(np.argmax(Q_values, axis=1))  # 최적의 정책(policy)

# gamma = 0, 0.9로 변경해서 테스트

# Q-value iteration에서 저장된 history를 그래프로 시각화
history_q_iter = np.array(history_q_iter)
print(history_q_iter.shape)  # (반복 횟수, state 개수, action 개수)
# x축은 반복 횟수. y축은 Q(s0, a0) 값만 plot
plt.plot(np.arange(50), history_q_iter[:, 0, 0])
plt.title('Q-Value Iteration Algorithm')
plt.show()

# Q-Learning Algorithm(Q-학습 알고리즘)
# Q[k+1] <- (1 - a) * Q[k](s,a) + a * (r + g * max Q[k](s',a'))
alpha0 = 0.05  # 학습률 초깃값
decay = 0.005  # 학습률 감쇠율 - 반복할 때마다 학습률을 줄여주기 위해서
gamma = 0.95  # 할인율
state = 0  # 상태의 초깃값
history_q_learn = []  # Q-learning 알고리즘으로 반복할 때마다 학습되는 Q값들을 저장.

# Q-Value 초기화
Q_values = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_values[state][actions] = 0.0


def explore_policy(state):
    """랜덤하게 행동을 선택"""
    return np.random.choice(possible_actions[state])


def step(state, action):
    prob = transition_probs[state][action]
    next_state = np.random.choice([0, 1, 2], p=prob)
    reward = rewards[state][action][next_state]
    return next_state, reward


# Q-learning
for iteration in range(10000):  # 전체 반복 횟수: 10,000
    history_q_learn.append(Q_values.copy())  # Q-value를 저장
    # 탐험 시작
    action = explore_policy(state)
    # 상태(state)에서 행동(action)을 해서, 다음 상태와 보상을 받음 - 게임 1 step 진행
    next_state, reward = step(state, action)
    # max Q(s',a') 가치를 계산
    next_q_value = np.max(Q_values[next_state])
    # 학습률 감쇠 - 반복할 수록 학습의 효과를 떨어뜨리기 위해서
    alpha = alpha0 / (1 + iteration * decay)
    # Q-Value 업데이트
    Q_values[state, action] *= (1 - alpha)
    Q_values[state, action] += alpha * (reward + gamma * next_q_value)
    # state를 다음 상태로 업데이트
    state = next_state

print(Q_values)
print(np.argmax(Q_values, axis=1))

# history에 저장된 Q-values들을 시각화
history_q_learn = np.array(history_q_learn)
print(history_q_learn.shape)  # (iteration, state, action)
plt.plot(np.arange(10000), history_q_learn[:, 0, 0])
plt.title('Q-Learning Algorithm')
plt.show()



