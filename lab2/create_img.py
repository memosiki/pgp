import random
import sys

name = 'big'
w = 10000
h = 10000
if __name__ == '__main__':
    # name = sys.argv[1]
    with open(f'{name}.data', 'wb') as file:
        a = w.to_bytes(4, 'little')
        file.write(a)
        a = h.to_bytes(4, 'little')
        file.write(a)

        for _ in range(w+1):
            for _ in range(h+1):
                num = random.randint(0x00_0000, 0xff_ffff) << 8
                file.write(num.to_bytes(4, 'big'))
