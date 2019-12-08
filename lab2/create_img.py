
if __name__ == '__main__':

    im = [
        [0x01020300, 0x04050600, 0x07080900],
        [0x09080700, 0x06050400, 0x03020100],
        [0x00000000, 0x14141400, 0x00000000],
    ]

    with open('in.data', 'wb') as file:
        w = len(im[0])
        h = len(im)
        a = w.to_bytes(4, 'little')
        file.write(a)
        a = h.to_bytes(4, 'little')
        file.write(a)

        for y in im:
            for x in y:
                file.write(x.to_bytes(4, 'big'))
