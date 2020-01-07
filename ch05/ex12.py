"""
ex12.py
ex11.py에서 저장한 pickle 파일을 읽어서,
파라미터(가중치/편향 행렬)들을 화면에 출력
"""
import pickle

with open('params.pickle', 'rb') as file:
    params = pickle.load(file)

for key, param in params.items():
    print(key, ':', param.shape)

