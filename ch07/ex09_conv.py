"""
ex09_conv.py
im2col 함수를 사용한 Convolution 구현
"""
import numpy as np
from common.util import im2col

if __name__ == '__main__':
    np.random.seed(115)

    # p.238 그림 7-11 참조
    # 가상의 이미지 데이터 1개를 생성
    # (n, c, h, w) = (이미지 개수, color-depth, height, width)
    x = np.random.randint(10, size=(1, 3, 7, 7))
    print(x, ', shape:', x.shape)

    # (3, 5, 5) 크기의 필터 1개 생성
    # (fn, c, fh, fw) = (필터 개수, color-dept, 필터 height, 필터 width)
    w = np.random.randint(5, size=(1, 3, 5, 5))
    print(w, ', shape:', w.shape)
    # 필터를 stride=1, padding=0으로 해서 convolution 연산
    # 필터를 1차원으로 펼침 -> c*fh*fw = 3 * 5 * 5 = 75

    # 이미지 데이터 x를 함수 im2col에 전달해서 2차원 배열로 변환
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col:', x_col.shape)
    # (9, 75) = (oh*ow, c*fh*fw)

    # 4차원 배열인 필터 w를 2차원 배열로 변환
    w_col = w.reshape(1, -1)  # row의 개수가 1, 모든 원소들은 column으로
    print('w_col:', w_col.shape)
    w_col = w_col.T
    print('w_col:', w_col.shape)

    # 2차원으로 변환된 이미지와 필터를 행렬 dot product 연산
    out = x_col.dot(w_col)
    print('out:', out.shape)

    # dot product의 결과를 (fn, oh, ow, ?) 형태로 reshape
    out = out.reshape(1, 3, 3, -1)
    print('out:', out.shape)  # (1, 3, 3, 1) = (n, oh, ow, fn)
    # color-depth 축(axis)가 두번째 축이 되도록 축의 위치를 변경.
    out = out.transpose(0, 3, 1, 2)
    print(out, 'shape:', out.shape)

    # p.238 그림 7-12, p.244 그림 7-19 참조
    # 가상으로 생성한 이미지 데이터 x와 2차원을 변환한 x_col를 사용
    # (3, 5, 5) 필터를 10개 생성 -> w.shape=(10, 3, 5, 5)
    w = np.random.randint(5, size=(10, 3, 5, 5))
    print('w shape:', w.shape)

    # w를 변형(reshape): (fn, c*fh*fw)
    w_col = w.reshape(10, -1)
    print('w_col shape:', w_col.shape)  # (10, 75)

    # x_col(9, 75) @ w_col.T(75, 10)  shape 확인
    out = x_col.dot(w_col.T)
    print('out shape:', out.shape)  # (9, 10)

    # dot 연산의 결과를 변형(reshape): (n, oh, ow, fn)
    out = out.reshape(1, 3, 3, 10)  # reshape(1, 3, 3, -1)
    print('out shape:', out.shape)

    # reshape된 결과에서 네번째 축이 두번째 축이 되도록 전치(transpose)
    out = out.transpose(0, 3, 1, 2)
    print('out shape:', out.shape)

    print()
    # 239 그림 7-13, p.244 그림 7-19 참조
    # (3, 7, 7) shape의 이미지 12개를 난수로 생성 -> (n, c, h, w) = (12, 3, 7, 7)
    x = np.random.randint(10, size=(12, 3, 7, 7))
    print(', shape:', x.shape)

    # (3, 5, 5) shape의 필터 10개 난수로 생성 -> (fn, c, fh, fw) = (10, 3, 5, 5)
    w = np.random.randint(5, size=(10, 3, 5, 5))
    print(', shape:', w.shape)

    # stride=1, padding=0일 때, output height, output width =?
    # oh = (h - fh + 2 * p) // s + 1 = (7 - 5) // 1 + 1 = 3
    # ow = (w - fw + 2 * p) // s + 1 = (7 - 5) // 1 + 1 = 3

    # 이미지 데이터 x를 im2col 함수를 사용해서 x_col로 변환 -> shape?
    x_col = im2col(x, filter_h=5, filter_w=5, stride=1, pad=0)
    print('x_col:', x_col.shape)
    # (108, 75) = (n * oh * ow, c * fh * fw)

    # 필터 w를 x_col과 dot 연산을 할 수 있도록 reshape & transpose: w_col -> shape?
    w_col = w.reshape(10, -1)
    print('w_col:', w_col.shape)   # (10, 75) = (fn, c * fh * fw)
    w_col = w_col.T
    print('w_col:', w_col.shape)   # (75, 10)

    # x_col @ w_col = (108, 75) @ (75, 10) = (108, 10)
    out = x_col.dot(w_col)
    print('out:', out.shape)

    # @ 연산의 결과를 reshape & transpose
    # (n * oh * ow, fn)
    out = out.reshape(12, 3, 3, 10)  # reshape(12, 3, 3, -1)
    print('out shape:', out.shape)
    out = out.transpose(0, 3, 1, 2)  # 인덱스를 사용하여 축 이동
    print('out shape:', out.shape)







