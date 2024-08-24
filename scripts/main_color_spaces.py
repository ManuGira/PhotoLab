import dataclasses
import enum
import cv2
import guibbon as gbn
import numpy as np
from photolab import utils as ut
from photolab import color_spaces as cs


class ColorSpaceApp:
    @dataclasses.dataclass()
    class Config:
        filter_type = 0
        channel0: int = 0
        channel1: int = 0
        channel2: int = 0
        color_space = cs.ColorSpace.BGR

    def __init__(self):
        self.win = gbn.create_window("Color Spaces")
        self.config = ColorSpaceApp.Config()

        self.color_space_list = [cs.ColorSpace.BGR, cs.ColorSpace.RGB, cs.ColorSpace.HLS, cs.ColorSpace.HSV, cs.ColorSpace.XYZ, cs.ColorSpace.LAB]
        self.win.create_radio_buttons("Color Space", [str(space).split(".")[-1] for space in self.color_space_list], self.on_change_color_space)

        self.slider_c0 = self.win.create_slider("00", range(255), self.on_change_channel0, initial_index=self.config.channel0)
        self.slider_c1 = self.win.create_slider("00", range(255), self.on_change_channel1, initial_index=self.config.channel1)
        self.slider_c2 = self.win.create_slider("00", range(255), self.on_change_channel2, initial_index=self.config.channel2)

        channel_names = cs.CHANNELS[self.config.color_space]
        self.slider_c0.name = channel_names[0]
        self.slider_c1.name = channel_names[1]
        self.slider_c2.name = channel_names[2]

        # self.slider_c0.name.set(channel_names[0])
        # self.slider_c1.name.set(channel_names[1])
        # self.slider_c2.name.set(channel_names[2])

        self.need_update = True

    def run(self):

        while self.win.is_alive:
            if self.need_update:
                self.result = self.update()
                self.need_update = False
            self.win.imshow(self.result)
            self.win.waitKeyEx(1)

    def update(self):

        c0 = self.config.channel0
        c1 = self.config.channel1
        c2 = self.config.channel2

        N = 256
        img = np.zeros((2 * N, 2 * N, 3), dtype=np.uint8)
        xs, ys, zs = np.meshgrid(range(N), range(N), range(1))

        img_01 = np.concatenate((ys, xs, zs + c2), axis=2)
        img[N:, N:] = img_01

        img_02 = np.concatenate((xs, zs + c1, ys), axis=2)
        img[N:0:-1, N:0:-1] = img_02

        img_12 = np.concatenate((zs + c0, xs, ys), axis=2)
        img[N:0:-1, N:] = img_12

        img = cs.ColorImage(img, self.config.color_space).to(cs.ColorSpace.BGR)

        # color square
        # img[N + N // 4:N + (3 * N) // 4, N // 4:(3 * N) // 4, :] = np.array([[[c0, c1, c2]]])

        # cursors
        img[:, N + c1, :] = 255
        img[N - c2, :, :] = 255
        img[N + c0, N:, :] = 255
        img[:N, N - c0, :] = 255

        img[N, :, :] = 0
        img[:, N, :] = 0
        return img

    def on_change_color_space(self, ind: int, val: str):
        self.config.color_space = self.color_space_list[ind]
        channel_names = cs.CHANNELS[self.config.color_space]
        self.slider_c0.name = channel_names[0]
        self.slider_c1.name = channel_names[1]
        self.slider_c2.name = channel_names[2]
        self.need_update = True

    def on_change_channel0(self, ind, val):
        self.config.channel0 = int(val)
        self.need_update = True

    def on_change_channel1(self, ind, val):
        self.config.channel1 = int(val)
        self.need_update = True

    def on_change_channel2(self, ind, val):
        self.config.channel2 = int(val)
        self.need_update = True


def main():
    dapp = ColorSpaceApp()
    dapp.run()


if __name__ == '__main__':
    main()
