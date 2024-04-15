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

DeuteranopiaMatrix = np.array([
    [0.625, 0.375, 0.000],
    [0.700, 0.300, 0.000],
    [0.000, 0.300, 0.700],
])

def apply_color_matrix(img, mat3x3):
    h, w = img.shape[:2]
    return (img.reshape(-1, 3) @ mat3x3.T).reshape(h, w, 3)

def cvtBRG_to_LMS(bgr):
    xyz = cv2.cvtColor(bgr, cv2.COLOR_BGR2XYZ).astype(float)/255
    return apply_color_matrix(xyz, HuntPointerEstevezMatrix)

def cvtLMS_to_BGR(lms):
    xyz = apply_color_matrix(lms, np.linalg.inv(HuntPointerEstevezMatrix))*255
    return cv2.cvtColor(xyz.astype(np.uint8), cv2.COLOR_XYZ2BGR)

def medium0_filter(img_bgr, strength):
    lms = cvtBRG_to_LMS(img_bgr)
    lms[:, :, 1] *= (lms[:, :, 1]*(1-strength)).astype(lms.dtype)
    return cvtLMS_to_BGR(lms)


def green0_filter(img_bgr, strength):
    res = img_bgr.copy()
    res[:, :, 1] = (res[:, :, 1]*(1-strength)).astype(res.dtype)
    return res

def deuteranopia_filter(img_bgr, strength):
    identity = np.identity(3, dtype=DeuteranopiaMatrix.dtype)
    mat = identity + strength*(DeuteranopiaMatrix-identity)
    result = apply_color_matrix(img_bgr, mat)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

class FilterType():
    NONE = "none"
    DEUTERANOPIA = "deuteranopia"
    GREEN0 = "green 0"
    MEDIUM0 = "medium 0"

class DaltonApp:
    @dataclasses.dataclass()
    class Config:
        filename: str = ""
        filter_type = FilterType.NONE
        strength = 1

    def __init__(self):
        self.win = gbn.create_window("Dalton")
        self.img_folder = "../../images/"
        images = ["Ishihara45.jpg", "daltonism_table.png", "baboon_512x512.png", "vermeer_758x640.jpg"]
        self.config = DaltonApp.Config(images[0])
        self.win.create_radio_buttons("filename", images, self.on_change_filename)
        self.win.create_radio_buttons("filter", [FilterType.NONE,FilterType.DEUTERANOPIA, FilterType.GREEN0, FilterType.MEDIUM0], self.on_change_filter_type)
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
        match self.config.filter_type:
            case FilterType.NONE:
                return img.copy()
            case FilterType.DEUTERANOPIA:
                return deuteranopia_filter(img, self.config.strength)
            case FilterType.GREEN0:
                return green0_filter(img, self.config.strength)
            case FilterType.MEDIUM0:
                return medium0_filter(img, self.config.strength)

    def on_change_filename(self, ind, val):
        self.config.filename = val
        self.need_update = True

    def on_change_filter_type(self, ind, val):
        self.config.filter_type = val
        self.need_update = True

    def on_change_strength(self, ind, val):
        self.config.strength = val
        self.need_update = True

def main():
    dapp = DaltonApp()
    dapp.run()


if __name__ == '__main__':
    main()
