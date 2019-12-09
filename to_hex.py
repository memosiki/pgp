def hex_chunk(data):
    return int.from_bytes(data, 'big')


with open('out.data', 'rb') as f:
    f.read(4 * 2)
    data = f.read(4)
    while True:
        for _ in range(3):
            print(f"{hex_chunk(data):08X}", end=' ')
            data = f.read(4)
        print()
        if not data:
            break
