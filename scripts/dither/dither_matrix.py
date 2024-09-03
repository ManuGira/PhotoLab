import numpy as np


def prime_factorization(x):
    for i in range(2, x):
        if (x // i) * i == x:
            return [i] + prime_factorization(x // i)
            break
    return [x]


def find_smaller_coprimes(x):
    """Find and return the list of all numbers coprimes with x and smaller than x"""
    coprimes = []
    x_factors = set(prime_factorization(x))
    for i in range(2, x):
        i_factors = set(prime_factorization(i))
        if len(i_factors.intersection(x_factors)) == 0:
            coprimes.append(i)

    return coprimes


M2_classic = np.array([
    [0, 2],
    [3, 1],
])

M2_diag = np.array([
    [0, 1],
    [3, 2],
])


def smiley(N=7):
    # 0 1
    # 2 3
    # 4 5

    # 6 7 8 9
    # 10 11 12 13
    # 14 15 16 17
    # 18 19 20 21 22

    # 23 24
    # 25 26

    # 27 28 29 30
    # 31 32 33 34
    # 35 36 37 38
    # 39 40 41 42

    # 43 44
    # 45 46
    # 47 48
    mat = np.array([
        [12,30,28,27,29,13,21],
        [37, 0,45,46, 1,41,17],
        [35,43,26,25,44,39,15],
        [36, 4,23,24, 5,40,14],
        [38,47, 2, 3,48,42,16],
        [10,34,32,31,33,11,20],
        [18, 8, 6, 7, 9,19,22],
    ])

    groups = [
        [0, 1], [2, 3], [4, 5],
        [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [23, 24], [25, 26],
        [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        [43, 44], [45, 46], [47, 48],
    ]

    mat = mat.astype(float)
    i = 0
    for group in groups:
        n = len(group)
        val = i + (n - 1) / 2
        for gval in group:
            mat[mat == gval] = val
        i += n

    return mat


def square(N=8):
    match N:
        case 4:
            return np.array([
                [0, 1, 2, 12],
                [7, 0, 3, 10],
                [6, 5, 4, 14],
                [13, 9, 11, 8],
            ])
        case 5:
            return np.array([
                [10, 16, 12, 21, 8],
                [20, 7, 0, 1, 19],
                [14, 6, 24, 2, 15],
                [18, 5, 4, 3, 22],
                [9, 23, 13, 17, 11],
            ])
        case 6:
            return np.array([
                [19,28,24,33,20,14],
                [32, 0, 1, 2,29,15],
                [23, 7,35, 3,25,16],
                [27, 6, 5, 4,34,17],
                [22,31,26,30,21,18],
                [ 8, 9,10,11,12,13],
            ])
        case 7:
            return np.array([
                [ 8, 9,10,11,12,13,14],
                [31,32,33,34,35,36,15],
                [30,47, 0, 1, 2,37,16],
                [29,46, 7,48, 3,38,17],
                [28,45, 6, 5, 4,39,18],
                [27,44,43,42,41,40,19],
                [26,25,24,23,22,21,20],
            ])
        case 8:
            return np.array([
                [ 8, 9,10,11,12,13,14,46],
                [31,47,48,49,50,51,15,45],
                [30,62, 0, 1, 2,52,16,44],
                [29,61, 7,63, 3,53,17,43],
                [28,60, 6, 5, 4,54,18,42],
                [27,59,58,57,56,55,19,41],
                [26,25,24,23,22,21,20,40],
                [32,33,34,35,36,37,38,39],
            ])

    if N < 4:
        return square(4)
    if N > 8:
        return square(7)



def get_order(weights):
    perm = np.argsort(weights.flatten())
    N = len(perm)
    res = np.zeros_like(weights).flatten()
    for i, x in enumerate(perm):
        res[x] = i
    res = res.reshape(weights.shape)
    return res


def spiral(N=8):
    xs, ys = np.meshgrid(range(N), range(N))
    xs = 2 * xs / (N - 1) - 1
    ys = 2 * ys / (N - 1) - 1
    rad = np.sqrt(xs ** 2 + ys ** 2)
    phi = np.arctan2(ys, xs)
    weights = rad + phi / 1000
    res = get_order(weights)
    return res


def hamming(N=8):
    xs, ys = np.meshgrid(range(N), range(N))
    xs = 2 * xs / (N - 1) - 1
    ys = 2 * ys / (N - 1) - 1
    rad = abs(xs) + abs(ys)
    phi = np.arctan2(ys, xs)
    weights = rad + phi / 1000
    res = get_order(weights)
    return res


def vert(N=8):
    assert N > 1
    best_c = 1

    if N > 2:
        coprimes = find_smaller_coprimes(N)

        golden_ratio = (5 ** 0.5 - 1) / 2
        smaller_dist = N
        for c in coprimes:
            dist = abs(c - golden_ratio * N)
            if dist < smaller_dist:
                smaller_dist = dist
                best_c = c

    xs, ys = np.meshgrid(range(N), range(N))
    xs = np.mod(xs * best_c, N)

    xs = 2 * xs / (N - 1)
    ys = 2 * ys / (N - 1)

    weights = abs(xs) + abs(ys) / 1000
    res = get_order(weights)
    return res


def horiz(N=8):
    res = vert(N)
    res = res.T
    return res


def diag(N=8):
    res = vert(N)

    for i in range(N):
        res[i, :] = np.roll(res[i, :], i)

    return res


def random(N=8):
    return np.random.permutation(N ** 2).reshape((N, N))


def main():
    import matplotlib.pyplot as plt
    plt.imshow(diag(8))
    plt.show()


if __name__ == '__main__':
    main()

# def get_matrix_by_name(name):
#     match name:
#         case "M2 classic":
#             return M2_classic
#         case "M2 diag":
#             return M2_diag
#
#     size, style = name.split()
#     size = int(size[1:])
#
#
#     match name:
#         case "spiral":
#             return spiral(N=size)
#         case "hamming":
#             return hamming(N=size)
