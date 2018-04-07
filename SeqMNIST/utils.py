def receptive_field(k, d):
    return d*(k-1)

if __name__ == "__main__":
    n_layers = 12
    r = 1
    for i in range(0, n_layers, 1):
        dil = int(1+ i*1.4)
        r_l = receptive_field(k=6, d=dil)
        print(dil)
        r += r_l
    print("total:", r*2-1)

    n_layers = 6
    r = 1
    for i in range(0, n_layers, 1):
        dil = int(1+ i*3)
        r_l = receptive_field(k=6, d=dil)
        print(dil)
        r += r_l
    print("total:", r*4-3)
