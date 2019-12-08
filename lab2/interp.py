import math

import numpy as np


def interp(im, i, j, h, w, h_new, w_new):
    ans = 0
    for pos in reversed(range(3)):
        # print(i, j, end=' : ')

        x = ((i + 0.5) * (w / w_new) - 0.5)
        y = ((j + 0.5) * (h / h_new) - 0.5)
        # print(x, y)
        x0 = math.floor(x)
        y0 = math.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1

        # print(x, x0)
        x0_ind = np.clip(x0, 0, w - 1)
        x1_ind = np.clip(x1, 0, w - 1)
        y0_ind = np.clip(y0, 0, h - 1)
        y1_ind = np.clip(y1, 0, h - 1)
        # x0_ind, x1_ind, y0_ind, y1_ind = x0, x1, y0, y1
        # if x0 == x1 == 0:
        #     x1_ind = x1 + 1
        #     x0 -= 1
        # if x0 == x1 == w - 1:
        #     x0_ind = x0 - 1
        #     x1 += 1
        # if y0 == y1 == 0:
        #     y1_ind = y1 + 1
        #     y0 -= 1
        # if y0 == y1 == h - 1:
        #     y0_ind = y0 - 1
        #     y1 += 1

        # if y0 == y1 or x0 == x1:
        #     print('??')

        Ia = (im[y0_ind][x0_ind] & (0xff << pos * 8)) >> pos * 8
        Ib = (im[y1_ind][x0_ind] & (0xff << pos * 8)) >> pos * 8
        Ic = (im[y0_ind][x1_ind] & (0xff << pos * 8)) >> pos * 8
        Id = (im[y1_ind][x1_ind] & (0xff << pos * 8)) >> pos * 8
        # if pos == 2:
            # print(Ia, Ib, Ic, Id)

        # print(y0, y, y1)
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        color = math.floor(wa * Ia + wb * Ib + wc * Ic + wd * Id)
        # держу в курсе: ((x1 - x0) * (y1 - y0)) == 1
        # print(color)
        ans <<= 1 * 8
        ans |= color
    return ans


if __name__ == '__main__':
    # im = [
    #     [0x010203, 0x040506, 0x070809],
    #     [0x090807, 0x060504, 0x030201],
    #     [0x000000, 0x141414, 0x000000],
    # ]
    im = [
        [0x010203, 0x040506, 0x070809],
        [0x090807, 0x060504, 0x030201],        [0x000000, 0x141414, 0x000000],
    ]
    w = len(im[0])
    h = len(im)
    new_w = 3
    new_h = 5

    for j in range(new_h):
        for i in range(new_w):
            a = interp(im, i, j, h, w, new_h, new_w)
            print(f"{a:06X}00", end=' ')
        print()
