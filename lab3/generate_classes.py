#! /home/konstanze/anaconda3/bin/python3
n_cls = 10
n_pxl = 10
wh = 100


if __name__ == '__main__':
    import numpy as np
    print(n_cls)
    for _ in range(n_cls):
        a = np.random.randint(0, wh-1, n_pxl);
        print(*a)