import dataclasses
import enum
import cv2
import guibbon as gbn
import numpy as np

# https://en.wikipedia.org/wiki/LMS_color_space
HuntPointerEstevezMatrix = np.array([
    [0.38971, 0.68898, -0.07868],  # L
    [-0.22981, 1.18340, 0.04641],  # M
    [0, 0, 1],  # S
])

HuntPointerEstevezMatrix /= np.sum(HuntPointerEstevezMatrix, axis=1)


def apply_color_matrix(img, mat3x3):
    h, w = img.shape[:2]
    return (img.reshape(-1, 3) @ mat3x3.T).reshape(h, w, 3)


def cvtBGR_to_LMS(bgr):
    xyz = cv2.cvtColor(bgr, cv2.COLOR_BGR2XYZ).astype(float) / 255
    return apply_color_matrix(xyz, HuntPointerEstevezMatrix)


def cvtLMS_to_BGR(lms):
    xyz = apply_color_matrix(lms, np.linalg.inv(HuntPointerEstevezMatrix)) * 255
    return cv2.cvtColor(xyz.astype(np.uint8), cv2.COLOR_XYZ2BGR)


class ColorSpaceApp:
    @dataclasses.dataclass()
    class Config:
        filter_type = 0
        channel0: int = 0
        channel1: int = 0
        channel2: int = 0

    def __init__(self):
        self.win = gbn.create_window("Color Spaces")
        self.config = ColorSpaceApp.Config()
        self.win.create_slider("Blue", range(255), self.on_change_channel0, initial_index=self.config.channel0)
        self.win.create_slider("Green", range(255), self.on_change_channel1, initial_index=self.config.channel1)
        self.win.create_slider("Red", range(255), self.on_change_channel2, initial_index=self.config.channel2)
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

        img[N+N//4:N+(3*N)//4, N//4:(3*N)//4, :] = np.array([[[c0, c1, c2]]])

        img[:, N + c1, :] = 255
        img[N - c2, :, :] = 255
        img[N+c0, N:, :] = 255
        img[:N, N-c0, :] = 255

        img[N, :, :] = 0
        img[:, N, :] = 0
        return img

    def on_change_filename(self, ind, val):
        self.config.filename = val
        self.need_update = True

    def on_change_filter_type(self, ind, val):
        self.config.filter_type = val
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
