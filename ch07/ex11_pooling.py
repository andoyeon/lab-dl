"""
ex11_polling.py
 4차원 데이터를 2차원으로 변환 후에 max pooling을 구현
"""
import numpy as np

from common.util import im2col

if __name__ == '__main__':
    np.random.seed(116)

    # 가상의 이미지 데이터 (c,h,w)=(3,4,4) 1개를 난수로 생성 -> (1,3,4,4)
    x = np.random.randint(10, size=(1, 3, 4, 4))
    print(x, 'shape:', x.shape)

    # 4차원 데이터를 2차원 ndarray로 변환
    col = im2col(x, filter_h=2, filter_w=2, stride=2, pad=0)
    print(col, 'shape:', col.shape)  # (4, 12)=(n*oh*ow, c*fh*fw)

    # max pooling: 채널별로 최대값을 찾음.
    # 채널별 최대값을 쉽게 찾기 위해서 2차원 배열의 shape을 변환
    # 변환된 행렬의 각 행에는, 채널별로 윈도우에 포함된 값들로만 이루어짐.
    col = col.reshape(-1, 2*2)  # (-1, fh*fw)
    print(col, 'shape:', col.shape)

    # 각 행(row)에서 최대값을 찾음.
    out = np.max(col, axis=1)
    print(out, 'shape:', out.shape)  # (12,)

    # 1차원 pooling의 결과를 4차원으로 변환
    out = out.reshape(1, 2, 2, 3)  # 원소 3개씩 2*3(2차원) 배열로 변환
    print(out)
    # (n, oh, ow, c) -> (n, c, oh, ow)
    out = out.transpose(0, 3, 1, 2)
    print(out)