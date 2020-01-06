"""
f(x, y, z) = (x + y)*z
x = -2, y = 5, z = -4에서의 df/dx, df/dy, df/dz의 값을
ex01에서 구현한 MultiplyLayer와 AddLayer 클래스를 이용해서 구하세요.
    q = x + y라 하면, dq/dx = 1, dq/dy = 1
    f = q * z 이므로, df/dq = z, df/dz = q
    위의 결과를 이용하면,
    df/dx = (df/dq)(dq/dx) = z
    df/dy = (df/dq)(dq/dy) = z

numerical_gradient 함수에서 계산된 결과와 비교
"""
from ch05.ex01_basic_layer import AddLayer, MultiplyLayer

x, y, z = -2, 5, -4

add_gate = AddLayer()
q = add_gate.forward(x, y)
print('q =', q)

mul_gate = MultiplyLayer()
f = mul_gate.forward(q, z)
print('f =', f)

dq, dz = mul_gate.backward(1)
print('dq =', dq)
print('dz =', dz)

dx, dy = add_gate.backward(dq)
print('dx =', dx)
print('dy =', dy)


def f(x, y, z):
    return (x + y) * z


h = 1e-12
dx = (f(-2 + h, 5, -4) - f(-2 - h, 5, -4)) / (2 * h)
print('df/dx =', dx)
dy = (f(-2, 5 + h, -4) - f(-2, 5 - h, -4)) / (2 * h)
print('df/dy =', dy)
dz = (f(-2, 5, -4 + h) - f(-2, 5, -4 - h)) / (2 * h)
print('df/dz =', dz)



