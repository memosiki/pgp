

m = [
[1, 1, 1, 1, 1, 1, 1],
[64, 32, 16, 8, 4, 2, 1],
[729, 243, 81, 27, 9, 3, 1],
[4096, 1024, 256, 64, 16, 4, 1],
[15625, 3125, 625, 125, 25, 5, 1],
[46656, 7776, 1296, 216, 36, 6, 1],
[117649, 16807, 2401, 343, 49, 7, 1],]

import numpy as np
# n = 10**3
# m = np.random.randint(-10,10,  (n, n))
a = np.linalg.det(m)
print(a)
