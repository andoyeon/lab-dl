"""
keras_functional_api.py
 Keras Functional API
"""
import numpy as np

from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import Input


# X(N, 64) -> Dense(32) -> ReLU -> Dense(32) -> ReLU -> Dense(10) -> Softmax
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

seq_model.summary()
# Param = X * W + b
# dense: 64 * 32 + 32 = 2080
# dense_1: 32 * 32 + 32 = 1056
# dense_2: 32 * 10 + 10 = 330

print()
# Keras의 함수형 API 기능을 사용해서 신경망 생성하는 방법
# Input 객체 생성
# -> 필요한 레이어 객체 생성 & 인스턴스 호출
# -> Model 객체를 생성
input_tensor = Input(shape=(64,))  # 입력 텐서의 shape을 결정
# 첫번째 은닉층(hidden layer) 생성 & 인스턴스 호출을 사용해서 입력 데이터 전달
# x = layers.Dense(32, activation='relu')(input_tensor)
dense1 = layers.Dense(32, activation='relu')
x = dense1(input_tensor)
