"""
ex01_gym.py
 OpenAI Gym 라이브러리 설치
 pip install gym
"""
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # gym 패키지 버전 확인
    print(gym.__version__)

    # gym 패키지의 환경(environment) 리스트 출력
    print(gym.envs.registry.all())

    # CartPole-v1 게임 환경을 생성
    env = gym.make('CartPole-v1')

    # 환경(environment)을 초기화
    obs = env.reset()   # observations(관찰, 관측)
    print(obs)
    # observation:[카트의 위치, 카트의 속도, 막대의 각도, 막대의 각속도]

    # 환경을 시각화
    env.render()

    # 환경(environment) 렌더링을 이미지 배열로 저장
    # img = env.render(mode='rgb_array')
    # print(img.shape)
    # matplotlib.pyplot을 이용한 이미지 출력
    # plt.imshow(img)
    # plt.show()

    # action: 게임을 실행, 게임 상태 변경
    # 가능한 액션의 개수
    print(env.action_space)  # Discrete(2)
    # CartPole 게임의 액션:
    # 0: 왼쪽 방향(-)으로 가속도를 줌.
    # 1: 오른쪽 방향(+)으로 가속도를 줌.

    action = 0  # 액션 종류를 설정
    obs, reward, done, info = env.step(action)   # 게임 상태 변경
    print(obs)

    action = 0  # 액션 종류를 설정
    obs, reward, done, info = env.step(action)   # 게임 상태 변경
    print(obs)

    # 사용했던 게임 환경 종료
    env.close()
