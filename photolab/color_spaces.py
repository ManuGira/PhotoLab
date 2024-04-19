import numpy as np
import cv2
import enum
import numba


def apply_color_matrix(img, mat3x3):
    h, w = img.shape[:2]
    return (img.reshape(-1, 3) @ mat3x3.T).reshape(h, w, 3)


def BGR_to_RGB_u8(bgr):
    assert bgr.dtype == np.uint8
    mat = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=np.uint8)
    res = apply_color_matrix(bgr, mat)
    assert res.dtype == np.uint8
    return res


def RGB_to_BGR_u8(rgb):
    assert rgb.dtype == np.uint8
    bgr = BGR_to_RGB_u8(rgb)
    assert bgr.dtype == np.uint8
    return bgr


def BGR_to_HLScube_u8(bgr):
    assert bgr.dtype == np.uint8
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    hls[:, :, 0] = (hls[:, :, 0].astype(np.float16) * (255 / 180)).astype(np.uint8)
    assert hls.dtype == np.uint8
    return hls


def HLScube_to_BGR_u8(hls):
    assert hls.dtype == np.uint8
    hls[:, :, 0] = (hls[:, :, 0].astype(np.float16) * (180 / 255)).astype(np.uint8)
    bgr = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    assert bgr.dtype == np.uint8
    return bgr


def BRG_to_HSVcube_u8(bgr):
    assert bgr.dtype == np.uint8
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float16) * (255 / 180)).astype(np.uint8)
    assert hsv.dtype == np.uint8
    return hsv


def HSVcube_to_BGR_u8(hsv):
    assert hsv.dtype == np.uint8
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float16) * (180 / 255)).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert bgr.dtype == np.uint8
    return bgr


def BGR_to_XYZ_u8(bgr):
    assert bgr.dtype == np.uint8
    res = cv2.cvtColor(bgr, cv2.COLOR_BGR2XYZ)
    assert res.dtype == np.uint8
    return res


def XYZ_to_BGR_u8(xyz):
    assert xyz.dtype == np.uint8
    bgr = cv2.cvtColor(xyz, cv2.COLOR_XYZ2BGR)
    assert bgr.dtype == np.uint8
    return bgr


def BGR_to_LAB_u8(bgr):
    assert bgr.dtype == np.uint8
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    assert lab.dtype == np.uint8
    return lab


def LAB_to_BGR_u8(lab):
    assert lab.dtype == np.uint8
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    assert bgr.dtype == np.uint8
    return bgr


class ColorSpace(enum.IntEnum):
    BGR = enum.auto()
    RGB = enum.auto()
    HLS = enum.auto()
    HSV = enum.auto()
    XYZ = enum.auto()
    LAB = enum.auto()


CHANNELS = {
    ColorSpace.BGR: ("Blue", "Green", "Red"),
    ColorSpace.RGB: ("Red", "Green", "Blue"),
    ColorSpace.HLS: ("Hue", "Light", "Saturation"),
    ColorSpace.HSV: ("Hue", "Saturation", "Value"),
    ColorSpace.XYZ: ("X", "Y", "Z"),
    ColorSpace.LAB: ("L", "A", "B"),
}


def convert_to_BGR_u8(img, from_space: ColorSpace):
    match from_space:
        case ColorSpace.BGR:
            return img
        case ColorSpace.RGB:
            return RGB_to_BGR_u8(img)
        case ColorSpace.XYZ:
            return XYZ_to_BGR_u8(img)
        case ColorSpace.HLS:
            return HLScube_to_BGR_u8(img)
        case ColorSpace.HSV:
            return HSVcube_to_BGR_u8(img)
        case ColorSpace.LAB:
            return LAB_to_BGR_u8(img)


def convert_from_BGR_u8(bgr, to_space: ColorSpace):
    match to_space:
        case ColorSpace.BGR:
            return bgr
        case ColorSpace.RGB:
            return BGR_to_RGB_u8(bgr)
        case ColorSpace.XYZ:
            return BGR_to_XYZ_u8(bgr)
        case ColorSpace.HLS:
            return BGR_to_HLScube_u8(bgr)
        case ColorSpace.HSV:
            return BRG_to_HSVcube_u8(bgr)
        case ColorSpace.LAB:
            return BGR_to_LAB_u8(bgr)


class ColorImage:
    def __init__(self, img, space: ColorSpace):
        self.img = convert_to_BGR_u8(img, from_space=space)

    def to(self, space: ColorSpace = ColorSpace.BGR):
        return convert_from_BGR_u8(self.img, to_space=space)


@numba.njit()
def apply_lut_numba(lut, img):
    res = np.zeros_like(img)
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            for c in range(3):
                res[y, x, c] = lut[img[y, x, 0], img[y, x, 1], img[y, x, 2], c]
    return res


class LUTConverter:
    """
    Experimental function that precompute a 255x255x255 look up table to convert between different color spaces.
    It is 10x slower than the opencv functions even when using numba
    """

    def __init__(self, from_space: ColorSpace, to_space: ColorSpace):
        r256 = np.arange(256, dtype=np.uint8)
        x, y, z = np.meshgrid(r256, r256, r256, indexing="ij")
        self.lut = np.concatenate((x.reshape(-1, 1, 1), y.reshape(-1, 1, 1), z.reshape(-1, 1, 1)), axis=2)
        self.lut = ColorImage(self.lut, from_space).to(to_space)
        self.lut = self.lut.reshape(256, 256, 256, 3)
        self.apply_lut_fast(np.zeros((1, 1, 1), dtype=np.uint8))

    def apply_lut_fast(self, img):
        return apply_lut_numba(self.lut, img)

    def apply_lut(self, img):
        return self.lut[img[:, :, 0], img[:, :, 1], img[:, :, 2]]


def main():
    import time
    import guibbon as gbn
    img = cv2.imread("../images/baboon_512x512.png")
    csc = LUTConverter(ColorSpace.BGR, ColorSpace.XYZ)
    gbn.imshow("ColorConversion", img)
    gbn.waitKeyEx(0)
    gbn.imshow("ColorConversion", cv2.cvtColor(img.copy(), cv2.COLOR_BGR2XYZ))
    gbn.waitKeyEx(0)
    gbn.imshow("ColorConversion", csc.apply_lut(img.copy()))
    gbn.waitKeyEx(0)

    def fps(dt):
        return 100 / (tock - tick)

    print("Time cv2: ", end="")
    tick = time.time()
    for k in range(100):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    tock = time.time()
    print(fps(tock - tick))

    print("Time mat mul: ", end="")
    tick = time.time()
    for k in range(100):
        img2 = BGR_to_RGB_u8(img)
    tock = time.time()
    print(fps(tock - tick))

    print("Time numba lut: ", end="")
    tick = time.time()
    for k in range(100):
        img2 = apply_lut_numba(csc.lut, img)
    tock = time.time()
    print(fps(tock - tick))

    print("Time np lut: ", end="")
    tick = time.time()
    for k in range(100):
        img2 = csc.apply_lut(img)
    tock = time.time()
    print(fps(tock - tick))


if __name__ == '__main__':
    main()
