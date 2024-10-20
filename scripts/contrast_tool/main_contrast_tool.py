import numpy as np
import numpy.typing as npt
import cv2
import guibbon as gbn
import matplotlib.pyplot as plt
import dataclasses
import photolab.utils as phutil

def make_lut_from_xtable(xtable):
    lut = np.array([0]*256, dtype=np.uint8)

    N = len(xtable)
    ytable = np.round(np.linspace(0, 255, N)).astype(np.uint8)

    lut[0:xtable[0]] = 0
    for i in range(N-1):
        xa = xtable[i]
        xb = xtable[i+1]
        ya = ytable[i]
        yb = ytable[i + 1]
        lut[xa:xb] = np.linspace(ya, yb, (xb-xa), endpoint=False)

    lut[xtable[-1]:] = 255
    return lut

def test_make_lut_from_xtable():
    lut = make_lut_from_xtable([0, 255])
    for k in range(256):
        assert lut[k] == k

    lut = make_lut_from_xtable([0, 10])
    assert lut[0] == 0
    assert 0 < lut[5] < 255
    for k in range(10, 256):
        assert lut[k] == 255

    lut = make_lut_from_xtable([10, 20])
    for k in range(10):
        assert lut[k] == 0

    assert 0 < lut[15] < 255
    for k in range(20, 256):
        assert lut[k] == 255

    lut = make_lut_from_xtable([10, 20, 50])
    for k in range(10):
        assert lut[k] == 0
    assert 0 < lut[15] < 127
    assert lut[20] == 128
    assert 127 < lut[25] < 255
    for k in range(50, 256):
        assert lut[k] == 255

def apply(img, low, mid, high):
    low = min(low, mid)
    high = max(mid, high)
    lut = make_lut_from_xtable([low, mid, high])
    return cv2.LUT(img, lut)

@dataclasses.dataclass()
class Config:
    low: int = 0
    mid: int = 128
    high: int = 255
    color: tuple[int, int, int] = (255, 255, 255)


def blend_max(img1, img2):
    res = img1.copy()
    mask = img1<img2
    res[mask] = img2[mask]
    return res


def colorize(img_gray, bgr):
    lut = np.zeros((256, 1, 3), dtype=float)

    for channel, value in enumerate(bgr):
        lut[:, 0, channel] = np.linspace(0, value, 256)
    lut = np.round(lut).astype(np.uint8)

    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    res = cv2.LUT(img_gray, lut)
    return res


class ContrastToolGUI:
    @dataclasses.dataclass()
    class Controls:
        low: gbn.SliderWidget
        mid: gbn.SliderWidget
        high: gbn.SliderWidget
        color: gbn.ColorPickerWidget

        def config_factory(self):
            config = Config(
                low=int(self.low.get_values()[self.low.get_index()]),
                mid=int(self.mid.get_values()[self.mid.get_index()]),
                high=int(self.high.get_values()[self.high.get_index()]),
                color=self.color.get_current_value()[::-1],
            )
            return config


    def __init__(self):
        self.win = gbn.create_window("Contrast")

        self.controls1 = ContrastToolGUI.Controls(
            self.win.create_slider("low1", range(256), self.set_update_flag, initial_index=0),
            self.win.create_slider("mid1", range(256), self.set_update_flag, initial_index=64),
            self.win.create_slider("high1", range(256), self.set_update_flag, initial_index=128),
            self.win.create_color_picker("color1", self.set_update_flag, initial_color_rgb=(255, 0, 255)),
        )

        self.controls2 = ContrastToolGUI.Controls(
            self.win.create_slider("low2", range(256), self.set_update_flag, initial_index=128),
            self.win.create_slider("mid2", range(256), self.set_update_flag, initial_index=196),
            self.win.create_slider("high2", range(256), self.set_update_flag, initial_index=255),
            self.win.create_color_picker("color2", self.set_update_flag, initial_color_rgb=(0, 255, 255)),
        )
        self.result: npt.NDArray
        self.is_update_needed = True

    def set_update_flag(self, *args, **kwargs):
        self.is_update_needed = True

    def update(self):
        config1 = self.controls1.config_factory()
        config2 = self.controls2.config_factory()

        img = cv2.imread("../../images/octopus.webp", cv2.IMREAD_GRAYSCALE)
        # img = np.clip(img.astype(int) + config.low, 0, 255).astype(np.uint8)
        img1 = 255 - apply(img, config1.low, config1.mid, config1.high)
        img2 = apply(img, config2.low, config2.mid, config2.high)

        img1 = colorize(img1, config1.color)
        img2 = colorize(img2, config2.color)

        self.result = blend_max(img1, img2)


    def show(self):
        while True:
            # self.is_update_needed = True
            if self.is_update_needed:
                self.update()
                self.win.imshow(self.result, mode=gbn.MODE.FIT, cv2_interpolation=cv2.INTER_NEAREST)
                self.is_update_needed = False
            gbn.waitKeyEx(1)


def main():
    test_make_lut_from_xtable()
    ContrastToolGUI().show()

if __name__ == '__main__':
    main()