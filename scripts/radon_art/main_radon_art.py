import dataclasses

import cv2
import numpy as np
import guibbon as gbn
import photolab as phl
import scipy
from photolab import math2d as m2


def prepare_image(img):
    H, W = img.shape[:2]
    D = max(H, W)
    img2 = np.zeros(shape=(D, D), dtype=np.uint8)
    shift = abs(H - W) // 2
    if H > W:
        img2[:, shift:shift + W] = img
    else:
        img2[shift:shift + H, :] = img

    mx, my = np.meshgrid(range(D), range(D))
    mx = mx / D * 2 - 1
    my = my / D * 2 - 1

    vignette = np.clip(10 * (1 - np.sqrt(mx ** 2 + my ** 2)), 0, 1)
    img2 = img2 * vignette / 255
    return (img2 * 255).astype(np.uint8)


def radon_transform(img, M, N):
    assert img.shape[0] == img.shape[1]
    assert img.dtype == np.uint8

    D = img.shape[0]
    radon = np.zeros((M, N), dtype=np.uint8)
    for i in range(N):
        theta = np.pi * i / N

        inv = np.linalg.inv

        TMat = m2.T((D / 2, D / 2))
        RMat = m2.R(theta)
        SMat = m2.S((M / D, M / D))

        transform_matrix = SMat @ TMat @ RMat @ inv(TMat)

        img_rot = cv2.warpAffine(img, M=transform_matrix[:2, :], dsize=(M, M), flags=cv2.INTER_LINEAR)

        radon[:, i] = np.mean(img_rot, axis=0)

    return radon


def random_threshold(img, coeff=0.001):
    H, W = img.shape[:2]
    thresh = (img.astype(float) / 255) * coeff
    rand_mat = np.random.rand(H, W)
    return (thresh > rand_mat).astype(np.uint8) * 255


def radon_backprojection(radon, D):
    result = np.zeros((D, D), dtype=np.int32)
    M, N = radon.shape[:2]
    for j in range(N):
        tmp = np.zeros_like(result, dtype=np.int32)
        theta = np.pi * j / N
        vx = D / 3 * np.sin(theta)
        vy = D / 3 * np.cos(theta)
        for i in range(M):
            radius = i * D / M - D / 2
            px_c = radius * np.cos(-theta) + D / 2
            py_c = radius * np.sin(-theta) + D / 2

            px_0, py_0 = np.round(px_c - vx).astype(int), np.round(py_c - vy).astype(int)
            px_1, py_1 = np.round(px_c + vx).astype(int), np.round(py_c + vy).astype(int)

            val = int(radon[i, j])
            print(val)
            cv2.line(tmp, (px_0, py_0), (px_1, py_1), (val,), thickness=1)
            result += tmp

    return (result.astype(float) / (M * N)).astype(np.uint8)


def binary_backprojection(bin_radon, D):
    result = np.zeros((D, D), dtype=np.uint8)
    radon_coords = np.argwhere(bin_radon == 255)
    M, N = bin_radon.shape[:2]
    radius = radon_coords[:, 0] * D / M - D / 2
    angles = radon_coords[:, 1] * np.pi / N

    px_c = radius * np.cos(-angles) + D / 2
    py_c = radius * np.sin(-angles) + D / 2
    vx = D / 3 * np.sin(angles)
    vy = D / 3 * np.cos(angles)

    px_0, py_0 = np.round(px_c - vx).astype(int), np.round(py_c - vy).astype(int)
    px_1, py_1 = np.round(px_c + vx).astype(int), np.round(py_c + vy).astype(int)

    for i in range(radon_coords.shape[0]):
        cv2.line(result, (px_0[i], py_0[i]), (px_1[i], py_1[i]), color=(255,), thickness=1)

    return result


def compute_radon_art():
    img = cv2.imread("../../images/vermeer_758x640.jpg", cv2.IMREAD_GRAYSCALE)
    img = prepare_image(img)
    radon = radon_transform(img, M=500, N=100)
    result = radon_backprojection(radon, 100)

    # bin_radon = random_threshold(radon)
    # result = binary_backprojection(bin_radon, 500)
    return result


class RadonArt:
    @dataclasses.dataclass()
    class Config:
        gain: float = 0

    def __init__(self):
        self.winname = "Radon Art"
        self.config = self.Config()
        win: gbn.Guibbon = gbn.create_window(self.winname)
        gain_range = [f"{v:.1f}" for v in np.linspace(0, 20, 21)]
        win.create_slider("gain", gain_range, self.onchange_gain_slider, initial_index=len(gain_range)//2)
        self.config.gain = 10
        
        self.result: Image_t
        self.is_update_needed: bool = True

    def onchange_gain_slider(self, i, gain):
        self.config.gain = float(gain)
        self.is_update_needed = True

    def update(self):
        self.result = compute_radon_art()

        self.result = np.clip(0, 255, self.result.astype(float) * 10**(self.config.gain/20)).astype(np.uint8)
        self.is_update_needed = False

    def show(self):
        while True:
            # self.is_update_needed = True
            if self.is_update_needed:
                self.update()
                gbn.imshow(self.winname, self.result, mode=gbn.MODE.FIT, cv2_interpolation=cv2.INTER_LINEAR)
                self.is_update_needed = False
            gbn.waitKeyEx(1)


def main():
    RadonArt().show()


if __name__ == '__main__':
    main()
