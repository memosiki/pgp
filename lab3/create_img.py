if __name__ == '__main__':

    im = [
        [0xA2DF4C00, 0xF7C9FE00, 0x9ED84500],
        [0xB4E85300, 0x99D14D00, 0x92DD5600],
        [0xA9E04C00, 0xF7D1FA00, 0xD4D0E900],
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
