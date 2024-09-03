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


def compute_dither_art(dither_mat, height: int = 200, gain_db: float = 0, invert: bool = False):
    img = cv2.imread("../../images/david_656x457.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../../images/ae86_563x1000.jpg", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../../images/ae86_543x1000.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("../../images/sphere_415x415.png", cv2.IMREAD_GRAYSCALE)
    img = photolab.utils.resize(img, new_height=height)

    img = np.clip(0, 255, img.astype(float) * 2 ** gain_db).astype(np.uint8)

    max_val = np.max(dither_mat.flatten())

    if invert:
        dither_mat = max_val - dither_mat

    dither_mat = np.round((dither_mat + 1).astype(float) * 255 / (max_val + 1)).astype(np.uint8)

    img = apply_dither_matrx(img, dither_mat)

    return img


class DitherArt:
    @dataclasses.dataclass()
    class Config:
        height: int = 0
        gain: float = 0
        dither_matrix_type: str = "hamming"
        dither_matrix_size: int = 0
        invert: bool = False
        matroll: int = 0
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
            "square",
            "smiley",
        ], self.onchange_dithermatrixtype)

        invert_check_button = win.create_check_button("invert", self.onchange_invert)
        matroll_slider = win.create_slider("mat roll", range(-5, 5), self.onchange_matroll_slider, initial_index=5)
        abberation_slider = win.create_slider("abberation", range(-15, 15), self.onchange_abberation_slider, initial_index=15)

        win.create_button("Save", self.onclick_save)
        win.create_button("Open Folder", self.onclick_openfolder)

        self.config = self.Config(
            height=height_slider.get_values()[height_slider.get_index()],
            gain=float(gain_slider.get_values()[gain_slider.get_index()]),
            dither_matrix_size=size_slider.get_values()[size_slider.get_index()],
            dither_matrix_type=matrix_type_options.get_current_selection()[1],
            invert=invert_check_button.get_current_value(),
            matroll=matroll_slider.get_values()[matroll_slider.get_index()],
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

    def onchange_invert(self, is_check: bool):
        self.config.invert = is_check
        self.is_update_needed = True

    def onchange_matroll_slider(self, i, val):
        self.config.matroll = val
        self.is_update_needed = True

    def onchange_abberation_slider(self, i, val):
        self.config.abberation = val
        self.is_update_needed = True

    def onclick_save(self):
        easy_save(self.result, self.savename)

    def onclick_openfolder(self):
        os.startfile(os.path.dirname(self.savename))

    def update(self):
        cfg = self.config

        match cfg.dither_matrix_type:
            case "spiral":
                mat = dither_matrix.spiral(N=cfg.dither_matrix_size)
            case "hamming":
                mat = dither_matrix.hamming(N=cfg.dither_matrix_size)
            case "random":
                mat = dither_matrix.random(N=cfg.dither_matrix_size)
            case "vert":
                mat = dither_matrix.vert(N=cfg.dither_matrix_size)
            case "diag":
                mat = dither_matrix.diag(N=cfg.dither_matrix_size)
            case "horiz":
                mat = dither_matrix.horiz(N=cfg.dither_matrix_size)
            case "square":
                mat = dither_matrix.square(N=cfg.dither_matrix_size)
            case "smiley":
                mat = dither_matrix.smiley(N=cfg.dither_matrix_size)
            case _:
                mat = dither_matrix.M2_classic

        shift_b = (cfg.matroll + 1) // 2
        shift_r = cfg.matroll // 2
        dither_b = compute_dither_art(np.roll(mat, -shift_b, axis=1), cfg.height, cfg.gain, cfg.invert)
        dither_g = compute_dither_art(mat, cfg.height, cfg.gain, cfg.invert)
        dither_r = compute_dither_art(np.roll(mat, shift_r, axis=1), cfg.height, cfg.gain, cfg.invert)

        shift_b = (cfg.abberation + 1) // 2
        shift_r = cfg.abberation // 2
        dither_b = np.roll(dither_b, -shift_b, axis=1)
        dither_g = dither_g
        dither_r = np.roll(dither_r, shift_r, axis=1)

        dither_b.shape += (1,)
        dither_g.shape += (1,)
        dither_r.shape += (1,)
        self.result = np.concatenate((dither_b, dither_g, dither_r), axis=2)

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
