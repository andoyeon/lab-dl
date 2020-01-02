"""
ex03.py
교재 p.160 그림 5-15의 빈칸 채우기.
apple = 100원, n_a = 2개
orange = 150원, n_o = 3개
tax = 1.1
라고 할 때, 전체 과일 구매 금액을 AddLayer와 MultiplyLayer를 사용해서 계산하세요.
df/dapple, df/dn_a, df/dorange, df/dn_o, df/dtax 값들도 각각 계산하세요.
"""
from ch05.ex01_basic_layer import MultiplyLayer, AddLayer

apple, n_a = 100, 2
orange, n_o = 150, 3
tax = 1.1

# Forward Propagation
apple_gate = MultiplyLayer()    # 뉴런 생성
apple_price = apple_gate.forward(apple, n_a)
print('apple price =', apple_price)

orange_gate = MultiplyLayer()    # 뉴런 생성
orange_price = orange_gate.forward(orange, n_o)
print('orange price =', orange_price)

app_org_gate = AddLayer()
all_price = app_org_gate.forward(apple_price, orange_price)
print('sum =', all_price)

tax_gate = MultiplyLayer()
total = tax_gate.forward(all_price, tax)
print('total =', total)

# Backward Propagation
delta = 1.0
dprice, dtax = tax_gate.backward(delta)
print('dprice =', dprice)
print('dtax =', dtax)   # df/dtax

dapp, dorg = app_org_gate.backward(dprice)
print('dapp =', dapp)
print('dorg =', dorg)

dapple, dn_a = apple_gate.backward(dapp)
print('dapple =', dapple)   # df/dapple
print('dn_a =', dn_a)   # df/dn_a

dorange, dn_o = orange_gate.backward(dorg)
print('dorange =', dorange)   # df/dorange
print('dn_o =', dn_o)   # df/dn_o
