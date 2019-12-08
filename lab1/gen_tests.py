from random import uniform

n = 2**25
name = 'big'
a = [0] * n
with open(f'lab1_test{name}', 'w') as f:
    f.write(str(n)+'\n')
    for i in range(n):
        val = uniform(-1000.0, 1000.0)
        f.write(f'{val:.10e} ')
        a[i] = val
    f.write('\n')

with open(f'lab1_test{name}.ans', 'w') as f:
    for elem in reversed(a):
        f.write(f'{elem:.10e} ')
    f.write('\n')
