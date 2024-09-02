import numpy as np

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


def diag(N=8):
    xs, ys = np.meshgrid(range(N), range(N))
    xs = 2 * xs / (N - 1)
    ys = 2 * ys / (N - 1)
    weights = abs(xs) + abs(ys)/1000
    res = get_order(weights)
    return res


def random(N=8):
    return np.random.permutation(N ** 2).reshape((N, N))

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
