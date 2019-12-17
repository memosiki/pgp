import warnings
import numpy as np

n = 10000


print(n)
a = np.random.uniform(0, 0.01, (n, n))
for i in range(n):
    for j in range(n):
        print(f"{a[i][j]:.10e}", end=' ')
    print()
warnings.warn('\n\n'+str(np.linalg.det(a)))
