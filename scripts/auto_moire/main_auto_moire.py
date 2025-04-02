import dataclasses
import os.path

import cv2
import guibbon as gbn
import numpy as np
import numpy.typing as npt

import photolab.utils as phutil

from numba import njit, vectorize
import time


class ModShift:
    @staticmethod
    def warmup():
        print("Start warmup")
        img12 = np.array([[0, 0]], dtype=np.uint8)
        img14 = np.array([[0, 0, 0, 0]], dtype=np.uint8)
        ModShift.accumulate_x(img12, 1)
        ModShift.apply_pattern(1, 2, 2, img12, img14)
        print("warmup done")

    @staticmethod
    @njit()
    def accumulate_x(img, mod):
        M, N = img.shape
        for i in range(M):
            for j in range(1, N):
                pix = img[i, j]
                pix_left = img[i, j - 1]
                img[i, j] = (pix + pix_left) % mod

    @staticmethod
    @njit()
    def apply_pattern(M: int, N: int, pattern_width: int, shift_map, result):
        depth = pattern_width // 2
        for i in range(M):
            y = i
            for j in range(N):
                for k in range(pattern_width):
                    x = j * pattern_width + k
                    result[y, x] = int(((k + shift_map[i, j]) % pattern_width) > depth)

    @staticmethod
    def apply(img, pattern_width) -> np.array:
        print("ModShift.apply()")

        shift_map = cv2.resize(img, None, None, fx=1 / pattern_width, fy=1, interpolation=cv2.INTER_AREA)
        M, N = shift_map.shape


        # we need: white -> no shift, black -> big shift. So we must invert the image
        shift_map = 255-shift_map

        # randomize first row to break vertical lines
        half = pattern_width // 2
        random_row = np.random.randint(0, 256, M // half + 1)
        random_row = random_row.repeat(half)[:M]
        shift_map[:, 0] = random_row

        # convert to int. Shift map correspond to amount of shift (roll) that needs to be applied on the pattern [0, 0, 0, 1, 1, 1] comparing to neighbor pattern on the left
        shift_map = (shift_map.astype(float) * pattern_width / 255).astype(np.uint8)

        # the accumulate func propagates each shift to neighbor pixel on the right. The shift is not absolute and not anymore relative to neighbor
        ModShift.accumulate_x(shift_map, pattern_width)

        HEIGHT = M
        WIDTH = N * pattern_width

        result = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        ModShift.apply_pattern(M, N, pattern_width, shift_map, result)

        return result


@dataclasses.dataclass()
class Config:
    shift: int = 0
    width: int = 10


class AutoMoireGUI:
    @dataclasses.dataclass()
    class Controls:
        width: gbn.SliderWidget
        shift: gbn.SliderWidget

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
            self.win.create_slider("pattern_width", range(2, 21, 2), self.set_prepare_flag, initial_index=5),
            self.win.create_slider("shift", range(-41, 41), self.set_update_flag, initial_index=41),
        )

        self.win.create_button("save", self.on_clicksave)
        self.prepare_result: npt.NDArray
        self.result: npt.NDArray
        self.is_prepare_needed = True
        self.is_update_needed = True

    def set_prepare_flag(self, *args, **kwargs):
        self.is_prepare_needed = True

    def set_update_flag(self, *args, **kwargs):
        self.is_update_needed = True

    def prepare(self):
        config = self.controls.config_factory()
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.prepare_result = ModShift.apply(img, config.width)

    def update(self):
        config = self.controls.config_factory()

        img2 = np.roll(self.prepare_result, shift=config.shift, axis=1)

        self.result = self.prepare_result * img2 * 255

    def show(self):
        while True:
            if self.is_prepare_needed:
                self.prepare()
                self.is_prepare_needed = False
                self.is_update_needed = True

            if self.is_update_needed:
                self.update()
                self.win.imshow(self.result, mode=gbn.MODE.FIT, cv2_interpolation=cv2.INTER_NEAREST)
                self.is_update_needed = False
            gbn.waitKeyEx(30)

    def on_clicksave(self):
        # res = make_alpha(self.result, [0, 0, 0])
        phutil.easy_save(self.result, self.savename)


def main():
    # AutoMoireGUI("../../images/octopus.webp").show()
    AutoMoireGUI("../../images/matcha dark.png").show()
    # AutoMoireGUI("../../images/david_656x457.jpg").show()


if __name__ == '__main__':
    ModShift.warmup()
    main()
