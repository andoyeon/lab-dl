import pandas as pd
import matplotlib.pyplot as plt

online_shoppers_intention = pd.read_csv('online_shoppers_intention.csv')
print(online_shoppers_intention.shape)

X = online_shoppers_intention.iloc[:, :-1]
y = online_shoppers_intention.iloc[:, -1:]

plt.

# 결측치 확인
print(online_shoppers_intention.isnull().sum())


