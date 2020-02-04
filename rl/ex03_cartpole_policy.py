"""
ex03_cartpole_policy.py
"""
import gym
import random
import numpy as np


def random_policy():
    """0 또는 1을 무작위로 생성해서 리턴"""
    return random.randint(0, 1)


def basic_policy(observation):
    """관측치(observation) 중에서 막대가 기울어진 각도에 따라서
    각도 > 0 인 경우에는, 오른쪽으로 가속도를 줌(action = 1)
    각도 < 0 인 경우에는, 왼쪽으로 가속도를 줌(action = 0)
    """
    angle = observation[2]
    if angle > 0:
        action = 1
    else:
        action = 0
    return action


if __name__ == '__main__':
    env = gym.make('CartPole-v1')  # 환경(environment) 생성

    max_episodes = 100  # 게임 실행 횟수
    # 1 에피소드 = 막대(pole)이 넘어지기 전까지(done == False)
    max_steps = 1000  # 1 에피소드에서 최대 반복 횟수

    total_rewards = []  # 에피소드가 끝날 때마다 얻은 보상을 저장할 리스트
    for episode in range(max_episodes):  # 게임 실행 횟수만큼 반복
        print(f'--- Episode #{episode + 1} ---')
        obs = env.reset()  # 게임 환경(environment) 초기화
        episode_reward = 0  # 1 에피소드에서 얻은 보상(reward)/점수
        for step in range(max_steps):  # 각 에피소드마다 최대 횟수만큼 반복
            env.render()  # 게임 화면 출력(렌더링)
            action = basic_policy(obs)  # random_policy()
            obs, reward, done, info = env.step(action)  # 게임 상태 변경
            episode_reward += reward   # 해당 에피소드의 보상을 계속 더함.
            if done:
                print(f'Episode finished after {step + 1} steps')
                break
        total_rewards.append(episode_reward)

    # 보상(점수)들의 리스트의 평균, 표준편차, 최대값, 최소값
    print(f'mean: {np.mean(total_rewards)}')
    print(f'std: {np.std(total_rewards)}')
    print(f'max: {np.max(total_rewards)}')
    print(f'min: {np.min(total_rewards)}')

    env.close()  # 게임 환경 종료
