import warnings
import numpy as np

n = 100


print(n)
a = np.random.uniform(-10, 10, (n, n))
for i in range(n):
    for j in range(n):
        print(f"{a[i][j]:.10e}", end=' ')
    print()
warnings.warn('\n\n'+str(np.linalg.det(a)))
