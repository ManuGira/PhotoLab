import dataclasses
import os.path

import cv2
import guibbon as gbn
import numpy as np
import numpy.typing as npt

import photolab.utils as phutil

from numba import njit, vectorize


@njit()
def accumulate_x(img, mod):
    M, N = img.shape
    for i in range(M):
        for j in range(1, N):
            pix = img[i, j]
            pix_left = img[i, j - 1]
            img[i, j] = (pix + pix_left) % mod
@njit()
def apply_pattern(M:int, N:int, pattern_width:int, shift_map, result):
    depth = pattern_width//2
    for i in range(M):
        y = i
        for j in range(N):
            for k in range(pattern_width):
                x = j * pattern_width + k
                result[y, x] = ((k + shift_map[i, j]) % pattern_width) // depth
                # np.roll(pattern, shift_map[i, j], axis=1)

def apply(img, pattern_width) -> np.array:
    depth = pattern_width // 2
    shift_map = cv2.resize(img, None, None, fx=1 / pattern_width, fy=1, interpolation=cv2.INTER_AREA)
    shift_map = (shift_map.astype(float) * depth / 255).astype(np.uint8)
    accumulate_x(shift_map, pattern_width)

    M, N = shift_map.shape
    HEIGHT = M
    WIDTH = N * pattern_width

    # for j in range(1, N):
    #     row = shift_map[:, j]
    #     prev_row = shift_map[:, j - 1]
    #
    #     shift_map[:, j] = np.mod(row + prev_row, depth)

    # accumulate_x(shift_map)

    pattern = np.zeros((1, pattern_width), dtype=np.uint8)
    pattern[:, depth:] = 1

    result = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    apply_pattern(M, N, pattern_width, shift_map, result)

    # result = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    # for i in range(M):
    #     y = i
    #     for j in range(N):
    #         x = j * pattern_width
    #         result[y:y + 1, x:x + pattern_width] = np.roll(pattern, shift_map[i, j], axis=1)

    return result


@dataclasses.dataclass()
class Config:
    shift: int = 0
    width: int = 10


class AutoMoireGUI:
    @dataclasses.dataclass()
    class Controls:
        shift: gbn.SliderWidget
        width: gbn.SliderWidget

        def config_factory(self):
            config = Config(
                shift=int(self.shift.get_values()[self.shift.get_index()]),
                width=int(self.width.get_values()[self.width.get_index()]),
            )
            return config

    def __init__(self, img_path):
        self.img_path = img_path
        name = os.path.basename(os.path.splitext(img_path)[0])
        self.savename = f"out/{name}.png"

        self.win = gbn.create_window("Auto Moiré")

        self.controls = AutoMoireGUI.Controls(
            self.win.create_slider("shift", range(-21, 21), self.set_update_flag, initial_index=21),
            self.win.create_slider("pattern_width", range(2, 21, 2), self.set_update_flag, initial_index=5),
        )

        self.win.create_button("save", self.on_clicksave)
        self.result: npt.NDArray
        self.is_update_needed = True

    def set_update_flag(self, *args, **kwargs):
        self.is_update_needed = True

    def update(self):
        config = self.controls.config_factory()

        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        # img = np.clip(img.astype(int) + config.shift, 0, 255).astype(np.uint8)
        img1 = apply(img, config.width)

        img2 = np.roll(img1, shift=config.shift, axis=1)

        self.result = img1 * img2 * 255

    def show(self):
        while True:
            # self.is_update_needed = True
            if self.is_update_needed:
                self.update()
                self.win.imshow(self.result, mode=gbn.MODE.FIT, cv2_interpolation=cv2.INTER_LINEAR)
                self.is_update_needed = False
            gbn.waitKeyEx(1)

    def on_clicksave(self):
        # res = make_alpha(self.result, [0, 0, 0])
        phutil.easy_save(self.result, self.savename)


def main():
    AutoMoireGUI("../../images/octopus.webp").show()


if __name__ == '__main__':
    main()
