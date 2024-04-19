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
        color_filter = cf.ColorFilter.NONE
        strength = 1

    def __init__(self):
        self.win = gbn.create_window("Dalton")
        self.img_folder = "../../images/"
        images = ["Ishihara45.jpg", "daltonism_table.png", "baboon_512x512.png", "vermeer_758x640.jpg"]
        self.config = DaltonApp.Config(images[0])
        self.win.create_radio_buttons("filename", images, self.on_change_filename)

        self.color_filter_list = [cf.ColorFilter.NONE, cf.ColorFilter.GRAYSCALE, cf.ColorFilter.SIMULATE_DEUTERANOPIA]
        self.win.create_radio_buttons("filter 1", [str(f).split(".")[-1] for f in self.color_filter_list], self.on_change_color_filter)
        self.win.create_slider("strength", np.linspace(0, 1, 31), self.on_change_strength, initial_index=30)
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
        return cf.apply_color_filter(img, self.config.color_filter)

    def on_change_filename(self, ind, val):
        self.config.filename = val
        self.need_update = True

    def on_change_color_filter(self, ind, val):
        self.config.color_filter = self.color_filter_list[ind]
        self.need_update = True

    def on_change_strength(self, ind, val):
        self.config.strength = val
        self.need_update = True


def main():
    dapp = DaltonApp()
    dapp.run()


if __name__ == '__main__':
    main()
