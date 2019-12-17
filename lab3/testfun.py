import numpy as np

a = np.random.rand(3, 3)


def swap(a, b):
    return b, a


def determ33(a):
    return a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) - \
           a[0][1] * (a[1][0] * a[2][2] - a[2][0] * a[1][2]) + \
           a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);


def inverse33(a):
    det = determ33(a);
    tmp = [[0. for _ in range(3)] for _ in range(3)]
    tmp[0][0] = 1 / det * (a[1][1] * a[2][2] - a[2][1] * a[1][2]);
    tmp[0][1] = -1 / det * (a[1][0] * a[2][2] - a[2][0] * a[1][2]);
    tmp[0][2] = 1 / det * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
    tmp[1][0] = -1 / det * (a[0][1] * a[2][2] - a[2][1] * a[0][2]);
    tmp[1][1] = 1 / det * (a[0][0] * a[2][2] - a[2][0] * a[0][2]);
    tmp[1][2] = -1 / det * (a[0][0] * a[2][1] - a[2][0] * a[0][1]);
    tmp[2][0] = 1 / det * (a[0][1] * a[1][2] - a[1][1] * a[0][2]);
    tmp[2][1] = -1 / det * (a[0][0] * a[1][2] - a[1][0] * a[0][2]);
    tmp[2][2] = 1 / det * (a[0][0] * a[1][1] - a[1][0] * a[0][1]);
    tmp[0][1], tmp[1][0] = swap(tmp[0][1], tmp[1][0]);
    tmp[0][2], tmp[2][0] = swap(tmp[0][2], tmp[2][0]);
    tmp[1][2], tmp[2][1] = swap(tmp[1][2], tmp[2][1]);
    return tmp

res1 = np.array(inverse33(a))
res2 = np.linalg.inv(a)
assert res1 == res2
print()
print()
