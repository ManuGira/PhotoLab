import dataclasses
import enum
import cv2
import guibbon as gbn
import numpy as np
from photolab import color_filters as cf


class DaltonApp:
    @dataclasses.dataclass()
    class Config:
        filename: str = ""
        color_filter_1: cf.ColorFilter = cf.ColorFilter.NONE
        shift: float = 0
        strength: float = 1
        color_filter_2: cf.ColorFilter = cf.ColorFilter.NONE

    def __init__(self):
        self.win = gbn.create_window("Dalton")
        self.img_folder = "../../images/"
        images = ["Ishihara45.jpg", "daltonism_table.png", "baboon_512x512.png", "vermeer_758x640.jpg", "hue_wheel_360.jpg", "hue_wheel_label.png"]
        self.config = DaltonApp.Config(images[0])
        self.win.create_radio_buttons("filename", images, self.on_change_filename)

        self.color_filter_1_list = [
            cf.ColorFilter.NONE,
            cf.ColorFilter.PROTANOPIA_LMS_CORRECTION,
            cf.ColorFilter.DEUTERANOPIA_LMS_CORRECTION,
            cf.ColorFilter.TRITANOPIA_LMS_CORRECTION,
        ]

        self.color_filter_2_list = [
            cf.ColorFilter.NONE,
            cf.ColorFilter.GRAYSCALE,
            cf.ColorFilter.SIMULATE_PROTANOPIA,
            cf.ColorFilter.SIMULATE_DEUTERANOPIA,
            cf.ColorFilter.SIMULATE_TRITANOPIA,
        ]

        self.win.create_radio_buttons("filter 1", [str(f).split(".")[-1] for f in self.color_filter_1_list], self.on_change_color_filter_1)
        self.win.create_slider("shift", np.linspace(-1, 1, 21), self.on_change_shift, initial_index=10)
        self.win.create_slider("strength", np.linspace(0, 5, 21), self.on_change_strength, initial_index=4)
        self.win.create_radio_buttons("filter 2", [str(f).split(".")[-1] for f in self.color_filter_2_list], self.on_change_color_filter_2)
        self.need_update = True

    def run(self):

        while self.win.is_alive:
            if self.need_update:
                self.result = self.update()
                self.need_update = False
            self.win.imshow(self.result)
            self.win.waitKeyEx(1)

    def update(self):
        img = cv2.imread(self.img_folder + self.config.filename)
        img = cf.apply_color_correction_filter(img, self.config.color_filter_1, self.config.shift, self.config.strength)
        return cf.apply_color_simulation_filter(img, self.config.color_filter_2)

    def on_change_filename(self, ind, val):
        self.config.filename = val
        self.need_update = True

    def on_change_color_filter_1(self, ind, val):
        self.config.color_filter_1 = self.color_filter_1_list[ind]
        self.need_update = True

    def on_change_color_filter_2(self, ind, val):
        self.config.color_filter_2 = self.color_filter_2_list[ind]
        self.need_update = True

    def on_change_shift(self, ind, val):
        self.config.shift = val
        self.need_update = True

    def on_change_strength(self, ind, val):
        self.config.strength = val
        self.need_update = True


def main():
    dapp = DaltonApp()
    dapp.run()


if __name__ == '__main__':
    main()
