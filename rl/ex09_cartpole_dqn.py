"""
DQN(Dep Q-Network)
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras


def epsilon_greedy_policy(model, state, epsilon=0.0):
    """DQN을 사용하는 action을 선택하는 정책.
    대부분(1 - epsilon 확률) 예상 Q-Value가 가장 큰 action을 선택하고,
    가끔(epsilon 확률) 랜덤하게 action을 선택"""
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


def sample_experiences(replay_memory, batch_size):
    """재현 메모리에서 경험을 샘플링:
    experience: [obs, actions, rewards, next_obs, dones]"""
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(model, env, state, epsilon, replay_memory):
    """DQN을 사용해서 게임 한 스텝을 진행.
     게임의 경험(obs, action, reward, next_state, done)을 재현 메모리에 저장."""
    action = epsilon_greedy_policy(model, state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


def training_step(replay_memory,
                  model,
                  n_outputs,
                  batch_size=32,
                  discount_rate=0.95,
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss_fn=keras.losses.mean_squared_error):
    """재현 메모리에서 경험들을 샘플링하고 모델을 학습시킴."""
    states, actions, rewards, next_states, dones = sample_experiences(replay_memory, batch_size)
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main():
    env = gym.make('CartPole-v1')
    input_shape = env.observation_space.shape  # (4,)
    n_outputs = env.action_space.n
    print(f'input_shape: {input_shape}, n_outputs: {n_outputs}')

    # DQN 모델
    model = keras.models.Sequential([
        keras.layers.Dense(32, activation='elu', input_shape=input_shape),
        keras.layers.Dense(32, activation='elu'),
        keras.layers.Dense(n_outputs)
    ])

    # 경험 데이터 사이의 상관관계를 감소시켜 학습을 향상시키기 위해, 재현 메모리를 사용.
    # (obs, action, reward, next_obs, done) 튜플을 학습의 매 스텝마다 재현 메모리에 저장.
    replay_memory = deque(maxlen=2000)

    batch_size = 32
    discount_rate = 0.95
    optimizer = keras.optimizers.Adam(lr=1e-3)
    loss_fn = keras.losses.mean_squared_error

    # 모델 학습
    env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    rewards = []  # plot으로 학습 결과를 시각화하기 위해 각 에피소드의 보상을 저장함.
    best_score = 0

    for episode in range(600):  # 600 에피소드(게임)을 하면서 학습
        obs = env.reset()
        for step in range(200):  # 1 에피소드(게임)에서 최대 스텝은 200회
            epsilon = max(1 - episode / 500, 0.01)  # epsilon 값은 1.0부터 시작해서 0.01까지 감소
            obs, reward, done, info = play_one_step(model, env, obs, epsilon, replay_memory)
            if done:
                break
        rewards.append(step)  # 에피소드가 끝나면 최종 보상을 저장함.
        if step > best_score:
            best_weights = model.get_weights()
            best_score = step
        print(f'\rEpisode: {episode}, Steps: {step+1}, eps: {epsilon}', end='')
        if episode > 50:
            training_step(replay_memory, model, n_outputs, batch_size)
    print()

    model.set_weights(best_weights)  # 최대 보상을 얻은 가중치들로 모델을 설정.

    # 학습 결과 시각화
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.show()

    # 완성된 모델로 새로운 게임 테스트
    env.seed(42)
    state = env.reset()
    env.render()
    for step in range(2000):
        action = epsilon_greedy_policy(model)
        state, reward, done, info = env.step(action)
        if done:
            break
        env.render()

    env.close()
    print(f'Finished after {step} steps')


if __name__ == '__main__':
    main()
