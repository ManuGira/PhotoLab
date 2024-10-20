import dataclasses
import numpy as np
import cv2

import guibbon as gbn

import photolab.utils
import photolab.utils_old
import dither_matrix
import os
import matrix_editor_widget
import numpy.typing as npt

from photolab.utils import easy_save


def make_alpha(img_bgr, color):
    h, w = img_bgr.shape[:2]
    alpha = np.zeros((h, w, 1), dtype=img_bgr.dtype) + 255
    color_match = np.all(img_bgr == color, axis=2)
    alpha[color_match] = 0

    img_bgra = np.concatenate((img_bgr, alpha), axis=2)
    return img_bgra


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


def preprocessing(img, black_offset: int, gain_db: float):
    img = (img.astype(float) - black_offset) * 2 ** gain_db
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


def compute_dither_art(img: npt.NDArray, dither_mat):
    max_val = np.max(dither_mat.flatten())
    dither_mat = np.round((dither_mat + 1).astype(float) * 255 / (max_val + 1)).astype(np.uint8)

    img = apply_dither_matrx(img, dither_mat)

    return img


def duplicate_with_rotation(matrix, rot_count):
    s = matrix.shape[0]

    result = np.zeros(shape=(2 * s, 2 * s), dtype=int)
    mat = matrix
    result[:s, :s] = mat

    for k in range(rot_count):
        mat = np.rot90(mat)
    result[s:, :s] = mat

    for k in range(rot_count):
        mat = np.rot90(mat)
    result[s:, s:] = mat

    for k in range(rot_count):
        mat = np.rot90(mat)
    result[:s, s:] = mat

    return result


def flip_4_quarters(matrix):
    s = matrix.shape[0]
    hs = s // 2

    matrix[hs:, :hs] = np.flipud(matrix[hs:, :hs])  # bot_left
    matrix[hs:, hs:] = np.rot90(matrix[hs:, hs:], 2)  # bot_right
    matrix[:hs, hs:] = np.fliplr(matrix[:hs, hs:])  # top_right
    return matrix


def rotate_4_quarters(matrix):
    s = matrix.shape[0]
    hs = s // 2

    matrix[hs:, :hs] = np.rot90(matrix[hs:, :hs], 1)
    matrix[hs:, hs:] = np.rot90(matrix[hs:, hs:], 2)
    matrix[:hs, hs:] = np.rot90(matrix[:hs, hs:], 3)
    return matrix


class DitherArt:
    @dataclasses.dataclass()
    class Config:
        height: int = 0
        black_offset: int = 0
        gain: float = 0
        dither_matrix_size: int = 0
        invert: bool = False
        matroll: int = 0
        abberation: int = 0
        dither_matrix_type: str = ""
        matrix: npt.NDArray = np.array([[0]])
        duplicate_rotate_count: int = 0
        duplicate_symmetry: bool = 0

    def __init__(self):
        self.winname = "Dither Art"
        self.savename = "out/dither.png"
        win: gbn.Guibbon = gbn.create_window(self.winname)
        height_range = (200 * 2 ** np.arange(-2, 3, dtype=float)).astype(int)
        height_slider = win.create_slider("res", height_range, self.onchange_height_slider, initial_index=len(height_range) // 2)

        black_offset_slider = win.create_slider("black offset", np.arange(255) - 127, self.onchange_black_offset_slider, initial_index=127)

        self.gain_range = np.linspace(-8, 8, 16 * 4 + 1)
        gain_range_str = [f"{v:.1f}" for v in self.gain_range]
        gain_slider = win.create_slider("gain", gain_range_str, self.onchange_gain_slider, initial_index=len(gain_range_str) // 2)

        size_slider = win.create_slider("matrix size", range(2, 10), self.onchange_size_slider, initial_index=0)

        self.matrix_type_options = win.create_radio_buttons("matrix type", [
            "hamming",
            "spiral",
            "random",
            "vert",
            "diag",
            "horiz",
            "square",
            "smiley",
            "custom",
        ], self.onchange_dithermatrixtype)

        self.matrix_editor = win.create_custom_widget(matrix_editor_widget.MatrixEditorWidget, "Matrix", 1, 1, np.array([[0]]), self.on_change_matrix_editor)

        self.duplicate_rotate_button_text = "rotate duplicate: %i"
        self.duplicate_rotate_button = win.create_button("", self.on_click_duplicate_rotate)
        duplicate_symmetry_check_button = win.create_check_button("Duplicate symmetry", self.on_click_duplicate_symmetry, initial_value=False)

        invert_check_button = win.create_check_button("invert", self.onchange_invert)
        matroll_slider = win.create_slider("mat roll", range(-12, 13), self.onchange_matroll_slider, initial_index=12)
        abberation_slider = win.create_slider("abberation", range(-15, 16), self.onchange_abberation_slider, initial_index=15)

        win.create_button("Save", self.onclick_save)
        win.create_button("Open Folder", self.onclick_openfolder)

        self.config = self.Config(
            height=height_slider.get_values()[height_slider.get_index()],
            black_offset=black_offset_slider.get_values()[black_offset_slider.get_index()],
            gain=float(gain_slider.get_values()[gain_slider.get_index()]),
            dither_matrix_size=size_slider.get_values()[size_slider.get_index()],
            invert=invert_check_button.get_current_value(),
            matroll=matroll_slider.get_values()[matroll_slider.get_index()],
            abberation=abberation_slider.get_values()[abberation_slider.get_index()],
            dither_matrix_type=self.matrix_type_options.get_current_selection()[1],
            matrix=self.matrix_editor.matrix,
            duplicate_rotate_count=duplicate_symmetry_check_button.get_current_value(),
        )
        self.duplicate_rotate_button.button["text"] = self.duplicate_rotate_button_text % self.config.duplicate_rotate_count

        self.update_matrix()

        self.result: Image_t
        self.is_update_needed: bool = True

    def update_matrix(self):
        cfg = self.config
        match cfg.dither_matrix_type:
            case "spiral":
                self.config.matrix = dither_matrix.spiral(N=cfg.dither_matrix_size)
            case "hamming":
                self.config.matrix = dither_matrix.hamming(N=cfg.dither_matrix_size)
            case "random":
                self.config.matrix = dither_matrix.random(N=cfg.dither_matrix_size)
            case "vert":
                self.config.matrix = dither_matrix.vert(N=cfg.dither_matrix_size)
            case "diag":
                self.config.matrix = dither_matrix.diag(N=cfg.dither_matrix_size)
            case "horiz":
                self.config.matrix = dither_matrix.horiz(N=cfg.dither_matrix_size)
            case "square":
                self.config.matrix = dither_matrix.square(N=cfg.dither_matrix_size)
            case "smiley":
                self.config.matrix = dither_matrix.smiley(N=cfg.dither_matrix_size)

        if cfg.dither_matrix_type == "custom":
            old_size = self.config.matrix.shape[0]
            new_size = cfg.dither_matrix_size
            if new_size < old_size:
                # crop to new size
                self.config.matrix = self.config.matrix[:new_size, :new_size]
                self.matrix_editor.set_matrix(self.config.matrix.astype(int))
            elif new_size > old_size:
                # 0-padding
                old_mat = self.config.matrix
                self.config.matrix = np.zeros(shape=(cfg.dither_matrix_size, cfg.dither_matrix_size), dtype=int)
                self.config.matrix[:old_size, :old_size] = old_mat
                self.matrix_editor.set_matrix(self.config.matrix.astype(int))
        else:
            self.matrix_editor.set_matrix(self.config.matrix.astype(int))

    def onchange_height_slider(self, i, val):
        self.config.height = int(val)
        self.is_update_needed = True

    def onchange_black_offset_slider(self, i, val):
        self.config.black_offset = int(val)
        self.is_update_needed = True

    def onchange_gain_slider(self, i, val):
        self.config.gain = self.gain_range[int(i)]
        self.is_update_needed = True

    def onchange_dithermatrixtype(self, i, dither_matrix_type):
        self.config.dither_matrix_type = dither_matrix_type
        self.update_matrix()
        self.is_update_needed = True

    def on_change_matrix_editor(self, matrix):
        self.matrix_type_options.select(index=self.matrix_type_options.get_options_list().index("custom"), trigger_callback=False)
        self.config.matrix = matrix
        self.is_update_needed = True

    def onchange_size_slider(self, i, val):
        self.config.dither_matrix_size = val
        self.update_matrix()
        self.is_update_needed = True

    def on_click_duplicate_rotate(self):
        self.config.duplicate_rotate_count = (self.config.duplicate_rotate_count + 1) % 4
        self.duplicate_rotate_button.button["text"] = self.duplicate_rotate_button_text % self.config.duplicate_rotate_count
        self.is_update_needed = True

    def on_click_duplicate_symmetry(self, value):
        self.config.duplicate_symmetry = value
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
        res = make_alpha(self.result, [0, 0, 0])
        easy_save(res, self.savename)

    def onclick_openfolder(self):
        os.startfile(os.path.dirname(self.savename))

    def update(self):
        cfg = self.config
        mat = np.tile(cfg.matrix, (2, 2))

        if self.config.duplicate_symmetry:
            mat = flip_4_quarters(mat)

        for k in range(cfg.duplicate_rotate_count):
            mat = rotate_4_quarters(mat)

        if cfg.invert:
            max_val = np.max(mat.flatten())
            mat = max_val - mat

        if True:
            # img = cv2.imread("../../images/david_656x457.jpg", cv2.IMREAD_GRAYSCALE)
            # img = cv2.imread("../../images/matcha dark.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.imread("../../images/octopus.webp", cv2.IMREAD_GRAYSCALE)
            # img = cv2.imread("../../images/256shades.png", cv2.IMREAD_GRAYSCALE)
            # img = cv2.imread("../../images/sphere_415x415.png", cv2.IMREAD_GRAYSCALE)
            # img[img == 255] = 0
            img = photolab.utils.resize(img, new_height=cfg.height)
        else:
            img = mat.copy().astype(float)
            img = (img * 255 / img.flatten().max()).astype(np.uint8)
            img = img.repeat(16, axis=0).repeat(16, axis=1)

        img = preprocessing(img, cfg.black_offset, cfg.gain)

        if len(img.shape) == 3 and img.shape[2] == 3:
            lut = (np.sqrt(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
            img = lut[img]
            img_b = img[:, :, 0]
            img_g = img[:, :, 1]
            img_r = img[:, :, 2]
        else:
            img_b = img
            img_g = img
            img_r = img

        shift_b = (cfg.matroll + 1) // 2
        shift_r = cfg.matroll // 2
        dither_b = compute_dither_art(img_b, np.roll(mat, -shift_b, axis=1))
        dither_g = compute_dither_art(img_g, mat)
        dither_r = compute_dither_art(img_r, np.roll(mat, shift_r, axis=1))

        shift_b = (cfg.abberation + 1) // 2
        shift_r = cfg.abberation // 2
        dither_b = np.roll(dither_b, -shift_b, axis=1)
        dither_g = dither_g
        dither_r = np.roll(dither_r, shift_r, axis=1)

        dither_b.shape += (1,)
        dither_g.shape += (1,)
        dither_r.shape += (1,)
        self.result = np.concatenate((dither_b, dither_g, dither_r), axis=2)

        # resize result
        h, w = self.result.shape[:2]
        size = min(h, w)
        factor = 800//size
        self.result = self.result.repeat(factor, axis=0).repeat(factor, axis=1)

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
