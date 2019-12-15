import warnings
import numpy as np

n = 300

print(n)
a = np.random.uniform(0, 1, (n, n))
for i in range(n):
    for j in range(n):
        print(a[i][j], end=' ')
    print()
warnings.warn(str(np.linalg.det(a)))
