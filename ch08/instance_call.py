"""
instance_call.py
"""
class Foo:
    def __init__(self, init_val=1):
        self.init_val = init_val


class Boo:
    def __init__(self, init_val=1):
        print('__init__ 호출')
        self.init_val = init_val

    def __call__(self, n):
        print('__call__ 호출')
        self.init_val *= n
        return self.init_val


if __name__ == '__main__':
    # Foo 클래스의 인스턴스를 생성 - 생성자 호출
    # instance: 주소값을 저장하고 있는 변수
    foo = Foo()
    print('init_val =', foo.init_val)
    # foo(100)  # 호출이 가능하지 않은 객체(not callable)

    # Boo 클래스의 인스턴스를 생성
    boo = Boo()
    print('boo.init_val =', boo.init_val)
    boo(5)
    # 인스턴스 호출: 인스턴스 이름을 마치 함수 이름처럼 사용하는 것.
    # 클래스에 정의된 __call__ 메소드를 호출하게 됨.
    # 만약, 클래스에서 __call__을 작성하지 않은 경우에는 인스턴스 호출을 사용할 수 없음.
    print('boo.init_val =', boo.init_val)

    # callabel: __call__ 메소드를 구현한 객체
    print('foo 호출 가능:', callable(foo))
    print('boo 호출 가능:', callable(boo))

    print()
    boo = Boo(2)  # 생성자 호출
    x = boo(2)  # 인스턴스(call 메소드) 호출
    print('x =', x)  # x = 4
    x = boo(x)
    print('x =', x)  # x = 16

    print()
    input = Boo(1)(5)
    print('input =', input)  # input = 5
    x = Boo(5)(input)
    print('x =', x)  # x = 25
    x = Boo(5)(x)
    print('x =', x)  # x = 125

