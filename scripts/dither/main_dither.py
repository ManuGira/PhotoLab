import dataclasses
import numpy as np
import cv2

import guibbon as gbn
import photolab.utils
import dither_matrix
import os


def easy_save(img, filename):
    dir = os.path.dirname(filename)
    filename_no_ext, ext = os.path.splitext(filename)

    os.makedirs(dir, exist_ok=True)

    n = 1
    while os.path.exists(filename):
        filename = f"{filename_no_ext}-{n:03}{ext}"
        n += 1

    cv2.imwrite(filename, img)



def apply_dither_matrx(img, mat):
    h, w = mat.shape[:2]
    H, W = img.shape[:2]
    xs, ys = np.meshgrid(range(W), range(H))
    xs = np.mod(xs, w).astype(np.float32)
    ys = np.mod(ys, w).astype(np.float32)
    dither_field = cv2.remap(mat, xs, ys, cv2.INTER_NEAREST)

    result = np.zeros_like(img)
    result[img >= dither_field] = 255
    return result


def compute_dither_art(dither_mat, height: int = 200, gain_db: float = 0):
    img = cv2.imread("../../images/david_656x457.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../../images/ae86_563x1000.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../../images/ae86_543x1000.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../../images/sphere_415x415.png", cv2.IMREAD_GRAYSCALE)
    img = photolab.utils.resize(img, new_height=height)

    img = np.clip(0, 255, img.astype(float) * 2 ** gain_db).astype(np.uint8)

    n = len(dither_mat.flatten())
    dither_mat = ((dither_mat + 1).astype(float) * 255 / (n + 1)).astype(np.uint8)
    img = apply_dither_matrx(img, dither_mat)

    return img


class DitherArt:
    @dataclasses.dataclass()
    class Config:
        height: int = 0
        gain: float = 0
        dither_matrix_type: str = "hamming"
        dither_matrix_size: int = 0
        abberation: int = 0

    def __init__(self):
        self.winname = "Dither Art"
        self.savename = "out/dither.png"
        win: gbn.Guibbon = gbn.create_window(self.winname)
        height_range = (200 * 2 ** np.arange(-2, 3, dtype=float)).astype(int)
        height_slider = win.create_slider("res", height_range, self.onchange_height_slider, initial_index=len(height_range) // 2)

        self.gain_range = np.linspace(-8, 8, 16 * 4 + 1)
        gain_range_str = [f"{v:.1f}" for v in self.gain_range]
        gain_slider = win.create_slider("gain", gain_range_str, self.onchange_gain_slider, initial_index=len(gain_range_str) // 2)

        size_slider = win.create_slider("matrix size", range(2, 10), self.onchange_size_slider, initial_index=0)

        matrix_type_options = win.create_radio_buttons("matrix type", [
            "classic",
            "hamming",
            "spiral",
            "random",
            "vert",
            "diag",
            "horiz",
        ], self.onchange_dithermatrixtype)

        abberation_slider = win.create_slider("abberation", range(-15, 15), self.onchange_abberation_slider, initial_index=15)

        win.create_button("Save", self.onclick_save)

        self.config = self.Config(
            height=height_slider.get_values()[height_slider.get_index()],
            gain=float(gain_slider.get_values()[gain_slider.get_index()]),
            dither_matrix_size=size_slider.get_values()[size_slider.get_index()],
            dither_matrix_type=matrix_type_options.get_current_selection()[1],
            abberation=abberation_slider.get_values()[abberation_slider.get_index()],
        )

        self.result: Image_t
        self.is_update_needed: bool = True

    def onchange_height_slider(self, i, val):
        self.config.height = int(val)
        self.is_update_needed = True

    def onchange_gain_slider(self, i, val):
        self.config.gain = self.gain_range[int(i)]
        self.is_update_needed = True

    def onchange_dithermatrixtype(self, i, name):
        self.config.dither_matrix_type = name
        self.is_update_needed = True

    def onchange_size_slider(self, i, val):
        self.config.dither_matrix_size = val
        self.is_update_needed = True

    def onchange_abberation_slider(self, i, val):
        self.config.abberation = val
        self.is_update_needed = True

    def onclick_save(self):
        easy_save(self.result, self.savename)

    def update(self):
        match self.config.dither_matrix_type:
            case "spiral":
                mat = dither_matrix.spiral(N=self.config.dither_matrix_size)
            case "hamming":
                mat = dither_matrix.hamming(N=self.config.dither_matrix_size)
            case "random":
                mat = dither_matrix.random(N=self.config.dither_matrix_size)
            case "vert":
                mat = dither_matrix.vert(N=self.config.dither_matrix_size)
            case "diag":
                mat = dither_matrix.diag(N=self.config.dither_matrix_size)
            case "horiz":
                mat = dither_matrix.horiz(N=self.config.dither_matrix_size)
            case _:
                mat = dither_matrix.M2_classic

        dither = compute_dither_art(mat, self.config.height, self.config.gain)

        self.result = np.repeat(np.zeros_like(dither, shape=dither.shape + (1,)), repeats=3, axis=2)
        shift_b = (self.config.abberation + 1) // 2
        shift_r = self.config.abberation // 2
        self.result[:, :, 0] = np.roll(dither, -shift_b, axis=1)
        self.result[:, :, 1] = dither
        self.result[:, :, 2] = np.roll(dither, shift_r, axis=1)

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
    DitherArt().show()


if __name__ == '__main__':
    main()
