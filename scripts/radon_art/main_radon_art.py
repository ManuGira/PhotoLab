import dataclasses

import cv2
import numpy as np
import guibbon as gbn
import photolab as phl
import scipy

import photolab.utils
import photolab.utils_old
from photolab import math2d as m2
import matplotlib.pyplot as plt


def fft2(img):
    imgf = np.fft.fftshift(np.fft.fft2(img))
    return imgf


def ifft2(imgf):
    H, W = imgf.shape[:2]
    assert H == W, "implemented for square images only"
    # assert H % 2 == 0, "implemented of odd dimension only"

    if H % 2 == 0:
        # imgf[H // 2 + 1:, :] = np.conj(imgf[:H // 2+1, :][::-1, ::-1])
        imgf = (imgf[1:, 1:] + np.conj(imgf[:0:-1, :0:-1])) / 2
    else:
        imgf[H // 2:, :] = np.conj(imgf[:H // 2+1, :][::-1, ::-1])
        # imgf = (imgf + np.conj(imgf[::-1, ::-1])) / 2

    img = np.fft.ifft2(np.fft.ifftshift(imgf))
    return img


def fft2_from_radon(radon):
    M, N = radon.shape[:2]
    fft1 = np.fft.fft(radon, axis=1)
    H, W = fft1.shape[:2]
    assert M == H
    assert N == W

    # plt.imshow(np.log(np.abs(fft1)))
    # plt.show()

    Hp, Wp = 2 * H, W // 2 + 1
    polar_fft = np.zeros_like(fft1, shape=(Hp, Wp))
    polar_fft[:H, :] = fft1[:, :Wp]
    polar_fft[H:, :1] = polar_fft[:H, :1]
    polar_fft[H:, 1:] = fft1[:, Wp:][:, ::-1]


    mx, my = np.meshgrid(np.arange(H) - H // 2, np.arange(H) - H // 2)
    mx = mx.astype(np.float32)
    my = my.astype(np.float32)
    mr = np.sqrt(mx ** 2 + my ** 2)
    mp = np.arctan2(my, mx) + np.pi
    my2 = mp * H / (2 * np.pi)
    mx2 = mr * Wp / (H / 2)

    # if True:
    #     xs, ys = np.meshgrid(np.arange(Wp), np.arange(Hp))
    #     xs = np.sin(xs/20*2*np.pi)*127+127
    #     ys = np.sin(ys/20*2*np.pi)*127+127
    #     plt.imshow(xs)
    #     plt.show()
    #     plt.imshow(cv2.remap(np.real(xs), mx2, my2, cv2.INTER_NEAREST))
    #     plt.show()
    #
    #     plt.imshow(ys)
    #     plt.show()
    #     plt.imshow(cv2.remap(ys, mx2, my2, cv2.INTER_NEAREST))
    #     plt.show()

    imgf_real = cv2.remap(np.real(polar_fft), mx2, my2, cv2.INTER_NEAREST)
    imgf_imag = cv2.remap(np.imag(polar_fft), mx2, my2, cv2.INTER_NEAREST)
    imgf = imgf_real + imgf_imag * complex(0, 1)
    # print(H, Hp, Wp)
    # plt.imshow(np.log(abs(imgf)))

    return imgf


def prepare_image(img):
    H, W = img.shape[:2]
    D = max(H, W)
    assert D % 2 == 0, "not implemented for non-odd dimension"

    # D += 1-D%2  # D must be odd
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
    for i in range(M):
        theta = np.pi * i / M

        TMat = m2.T((D / 2, D / 2))
        RMat = m2.R(theta)
        SMat = m2.S((N / D, N / D))

        inv = np.linalg.inv
        transform_matrix = SMat @ TMat @ RMat @ inv(TMat)

        img_rot = cv2.warpAffine(img, M=transform_matrix[:2, :], dsize=(N, N), flags=cv2.INTER_LINEAR)

        radon[i, :] = np.mean(img_rot, axis=0)

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
    img = photolab.utils.resize(img, new_height=51)

    H, W = img.shape[:2]
    assert H == W
    M = int(round(np.pi * H))
    N = H
    M += M % 2
    N += N % 2 - 1

    # imgf = fft2(img)
    # imgpf_real = cv2.warpPolar(np.real(imgf), dsize=(M//2+1, N), center=(H//2, H//2), maxRadius=H//2, flags=cv2.WARP_POLAR_LINEAR)
    # imgpf_imag = cv2.warpPolar(np.imag(imgf), dsize=(M//2+1, N), center=(H//2, H//2), maxRadius=H//2, flags=cv2.WARP_POLAR_LINEAR)
    # imgf_real = cv2.warpPolar(imgpf_real, dsize=(H, H), center=(H//2, H//2), maxRadius=H//2, flags=cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
    # imgf_imag = cv2.warpPolar(imgpf_imag, dsize=(H, H), center=(H//2, H//2), maxRadius=H//2, flags=cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
    # imgf = imgf_real + imgf_imag*complex(0, 1)
    # img = np.abs(ifft2(imgf))

    # return abs(fft2(img))
    radon = radon_transform(img, M=M, N=N)
    imgf = fft2_from_radon(radon)

    result = ifft2(imgf)
    return result


class RadonArt:
    @dataclasses.dataclass()
    class Config:
        gain: float = 0

    def __init__(self):
        self.winname = "Radon Art"
        win: gbn.Guibbon = gbn.create_window(self.winname)
        gain_range = [f"{v:.1f}" for v in np.linspace(-40, 40, 41)]

        gain_slider = win.create_slider("gain", gain_range, self.onchange_gain_slider, initial_index=len(gain_range) // 2)
        self.config = self.Config(
            gain=float(gain_range[gain_slider.get_index()])
        )

        self.result: Image_t
        self.is_update_needed: bool = True

    def onchange_gain_slider(self, i, gain):
        self.config.gain = float(gain)
        self.is_update_needed = True

    def update(self):
        self.result = compute_radon_art()

        self.result = np.clip(0, 255, self.result.astype(float) * 10 ** (self.config.gain / 20)).astype(np.uint8)
        self.is_update_needed = False

    def show(self):
        while True:
            # self.is_update_needed = True
            if self.is_update_needed:
                self.update()
                gbn.imshow(self.winname, self.result, mode=gbn.MODE.FIT, cv2_interpolation=cv2.INTER_NEAREST)
                self.is_update_needed = False
            gbn.waitKeyEx(1)


def main():
    RadonArt().show()


if __name__ == '__main__':
    main()
