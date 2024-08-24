import cv2
from photolab import utils
import guibbon as gbn
import numpy as np
import time

update_needed = False

GAIN_DB = 0


def on_change_gain(index, val):
    global GAIN_DB, update_needed
    GAIN_DB = val
    update_needed = True


def rgb_gain(img, gain: float):
    dst = img.astype(float)
    dst *= gain
    dst[dst > 255] = 255
    return dst.astype(np.uint8)


def hls_gain(img, gain: float):
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    dst_l = dst[:, :, 1].astype(float)
    dst_l *= gain
    dst_l[dst_l > 255] = 255
    dst[:, :, 1] = dst_l.astype(np.uint8)
    dst = cv2.cvtColor(dst, cv2.COLOR_HLS2BGR)
    return dst


def mat_gain(img, gain: float):
    cos30 = np.cos(np.pi / 6)  # sqrt(3)/2
    sin30 = np.sin(np.pi / 6)  # 1/2

    base_mat = np.array([
        [-cos30, cos30, 0],
        [-sin30, -sin30, 1],
        [1 / 9, 6 / 9, 2 / 9],
    ], dtype=float)

    gain_mat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, gain]
    ], dtype=float)

    mat = np.linalg.inv(base_mat) @ gain_mat @ base_mat

    h, w = img.shape[:2]
    pixels = img.reshape((h * w, 3)).transpose().astype(float)
    pixels = mat @ pixels
    pixels[pixels > 255] = 255
    pixels[pixels < 0] = 0
    dst = pixels.astype(np.uint8).transpose().reshape((h, w, 3))
    return dst


def array_of_img(H, W, imgs):
    h, w = imgs[0].shape[:2]
    Hh = H * h
    Ww = W * w
    dst = np.zeros(shape=(Hh, Ww, 3), dtype=np.uint8)
    k = 0
    for i in range(H):
        for j in range(W):
            if k >= len(imgs):
                return dst
            dst[i * h:(i + 1) * h, j * w:(j + 1) * w] = imgs[k]
            k += 1
    return dst


def main():
    global update_needed, GAIN_DB
    filepath = "../images/baboon_512x512.png"
    img_src = cv2.imread(filepath)
    if img_src is None:
        raise FileNotFoundError(filepath)
    winname = "brightness"
    gbn.namedWindow(winname)

    N = 10
    gbn.create_slider(winname, "gain dB", range(-N, N + 1), on_change_gain, N + 1)

    while True:
        gain = 10 ** (GAIN_DB / 20)
        print(gain)
        imgs = []
        dts = []
        tic = time.time()
        imgs.append(rgb_gain(img_src, gain))
        toc = time.time()
        dts.append((toc - tic) * 1000)
        tic = toc

        imgs.append(hls_gain(img_src, gain))
        toc = time.time()
        dts.append((toc - tic) * 1000)
        tic = toc

        imgs.append(mat_gain(img_src, gain))
        toc = time.time()
        dts.append((toc - tic) * 1000)
        tic = toc

        img_dst = array_of_img(2, 2, imgs)
        print(f"{dts}")
        gbn.imshow(winname, img_dst)
        update_needed = False
        while not update_needed:
            gbn.waitKeyEx(50)


if __name__ == '__main__':
    main()
