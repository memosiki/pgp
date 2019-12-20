import random
import sys

name = 'big'
w = 1000
h = 1000
if __name__ == '__main__':
    # name = sys.argv[1]
    with open(f'{name}.data', 'wb') as file:
        a = w.to_bytes(4, 'little')
        file.write(a)
        a = h.to_bytes(4, 'little')
        file.write(a)

        for y in range(w):
            for x in range(h):
                num = random.randint(0x00_0000, 0xff_ffff) << 8
                file.write(num.to_bytes(4, 'big'))
