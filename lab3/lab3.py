import numpy as np


#
def det22(a):
    return a[0][0] * a[1][1] - a[1][0] * a[0][1]


def det33(a):
    return a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2]) - \
           a[0][1] * (a[1][0] * a[2][2] - a[2][0] * a[1][2]) + \
           a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1])


def inv33(a):
    ret = np.zeros((3, 3))
    det = det33(a)
    ret[0][0] = 1 / det * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
    ret[0][1] = - 1 / det * (a[1][0] * a[2][2] - a[2][0] * a[1][2])
    ret[0][2] = 1 / det * (a[1][0] * a[2][1] - a[2][0] * a[1][1])
    ret[1][0] = -1 / det * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
    ret[1][1] = 1 / det * (a[0][0] * a[2][2] - a[2][0] * a[0][2])
    ret[1][2] = -1 / det * (a[0][0] * a[2][1] - a[2][0] * a[0][1])
    ret[2][0] = 1 / det * (a[0][1] * a[1][2] - a[1][1] * a[0][2])
    ret[2][1] = - 1 / det * (a[0][0] * a[1][2] - a[1][0] * a[0][2])
    ret[2][2] = -1 / det * (a[0][0] * a[1][1] - a[1][1] * a[0][1])

    ret[0][1], ret[1][0] = ret[1][0], ret[0][1]
    ret[0][2], ret[2][0] = ret[2][0], ret[0][2]
    ret[1][2], ret[2][1] = ret[2][1], ret[1][2]
    return ret

# a = [[894.028, -9.11111, 1285.81, ], [-9.11111, 119.639, -171.917, ], [1290.78, -171.917, 2187.44
#                                                                        ]]

# def dot13to33(a, b):
#
#
# a = np.array([[1, 2, 3], [10, 4, 5], [6, 7, 9]])
# print(np.linalg.inv(a))
# print(inv33(a))
# exit(0)


# print(inv33(a))
# exit(0)


def inverseof33matrix(m):
    pass


def ppx(px):
    args = [px[i][0] for i in range(len(px))]
    return "{:02X}{:02X}{:02X}0".format(*args)


im = np.array([
    [[[0xA2], [0xDF], [0x4C]], [[0xF7], [0xC9], [0xFE]], [[0x9E], [0xD8], [0x45]]],
    [[[0xB4], [0xE8], [0x53]], [[0x99], [0xD1], [0x4D]], [[0x92], [0xDD], [0x56]]],
    [[[0xA9], [0xE0], [0x4C]], [[0xF7], [0xD1], [0xFA]], [[0xD4], [0xD0], [0xE9]]],
])

gr1 = iter([1, 2, 1, 0, 2, 2, 2, 1])
gr2 = iter([0, 0, 0, 1, 1, 1, 2, 0])
groups = [gr1, gr2]
avg = []
for j in range(len(groups)):
    gr = groups[j]
    summ = np.zeros((3, 1))
    npj = 4
    for i in range(npj):
        y = next(gr)
        x = next(gr)
        p = im[x, y]
        summ += p
    # print(summ)
    avg.append(summ / npj)
cov = []
gr1 = iter([1, 2, 1, 0, 2, 2, 2, 1])
gr2 = iter([0, 0, 0, 1, 1, 1, 2, 0])
groups = [gr1, gr2]
for j in range(len(groups)):
    gr = groups[j]
    summ = np.zeros((3, 3))
    npj = 4
    for i in range(npj):
        y = next(gr)
        x = next(gr)
        p = im[x, y]
        summ += (p - avg[j]) @ np.transpose(p - avg[j])
    cov.append(summ / (npj - 1))
    # print('inv', np.linalg.inv(cov[j]))
    # print(cov[j])
# print(*avg)

cov = [np.linalg.inv(c) for c in cov]
# print(*cov, sep='\n')

for row in im:
    for px in row:
        a = np.array([
            (-(
                    np.transpose(px - avg[j]) @ cov[j] @ (px - avg[j])
            )[0][0])
            for j in range(2)
        ]).argmax()
        for j in range(2):
            print(np.transpose(px - avg[j]))
        print(ppx(px), a, sep='', end=' ')
    print()
