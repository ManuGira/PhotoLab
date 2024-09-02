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
