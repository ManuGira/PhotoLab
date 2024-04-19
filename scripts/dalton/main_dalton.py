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
        color_filter_1 = cf.ColorFilter.NONE
        color_filter_2 = cf.ColorFilter.NONE
        # strength = 1

    def __init__(self):
        self.win = gbn.create_window("Dalton")
        self.img_folder = "../../images/"
        images = ["Ishihara45.jpg", "daltonism_table.png", "baboon_512x512.png", "vermeer_758x640.jpg"]
        self.config = DaltonApp.Config(images[0])
        self.win.create_radio_buttons("filename", images, self.on_change_filename)

        self.color_filter_1_list = [
            cf.ColorFilter.NONE,
            cf.ColorFilter.FIX_PROTANOPIA,
            cf.ColorFilter.FIX_DEUTERANOPIA,
            cf.ColorFilter.FIX_TRITANOPIA,
        ]

        self.color_filter_2_list = [
            cf.ColorFilter.NONE,
            cf.ColorFilter.GRAYSCALE,
            cf.ColorFilter.SIMULATE_PROTANOPIA,
            cf.ColorFilter.SIMULATE_DEUTERANOPIA,
            cf.ColorFilter.SIMULATE_TRITANOPIA,
        ]

        self.win.create_radio_buttons("filter 1", [str(f).split(".")[-1] for f in self.color_filter_1_list], self.on_change_color_filter_1)
        self.win.create_radio_buttons("filter 2", [str(f).split(".")[-1] for f in self.color_filter_2_list], self.on_change_color_filter_2)
        # self.win.create_slider("strength", np.linspace(0, 1, 31), self.on_change_strength, initial_index=30)
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
        img = cf.apply_color_filter(img, self.config.color_filter_1)
        return cf.apply_color_filter(img, self.config.color_filter_2)

    def on_change_filename(self, ind, val):
        self.config.filename = val
        self.need_update = True

    def on_change_color_filter_1(self, ind, val):
        self.config.color_filter_1 = self.color_filter_1_list[ind]
        self.need_update = True

    def on_change_color_filter_2(self, ind, val):
        self.config.color_filter_2 = self.color_filter_2_list[ind]
        self.need_update = True

    # def on_change_strength(self, ind, val):
    #     self.config.strength = val
    #     self.need_update = True


def main():
    dapp = DaltonApp()
    dapp.run()


if __name__ == '__main__':
    main()
