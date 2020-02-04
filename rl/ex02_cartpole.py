"""
ex02_cartpole.py
"""
import gym
import numpy as np

if __name__ == '__main__':
    # 게임 environment를 생성
    env = gym.make('CartPole-v1')

    # 게임 환경 초기화
    obs = env.reset()
    # 초기화 화면 출력
    env.render()
    print(obs)

    max_steps = 1000  # 최대 반복 횟수
    # for문 반복할 때마다 action을 값이 0 또는 1을 랜덤하게 선택하도록
    # done 값이 True이면 for loop을 종료
    # 몇 번 step만에 게임이 종료됐는 지 출력
    for t in range(max_steps):
        action = np.random.randint(0, 1)  # 게임 액션 설정
        obs, reward, done, info = env.step(action)  # 게임 진행
        env.render()   # 게임 환경 화면 출력
        # print(obs)
        print(f'reward: {reward}, done: {done}, info: {info}')
        if done:
            print(f'----- Finished after {t + 1} steps -----')
            break

    env.close()  # 게임 환경 종료
