import enum

import numpy as np
import cv2
from photolab import color_spaces as cs

BGR_to_LMS_matrix = np.array([
    # BGR to LMS matrix
    [4.11935, 43.5161, 17.884],
    [3.86714, 27.1554, 3.45565],
    [1.46709, 0.184309, 0.0299566],
])

LMS_Deuteranopia_matrix = np.array([
    [1, 0, 0],
    [0.49421, 0, 1.24827],
    [0, 0, 1],
])
BGR_Deuteranopia_matrix = np.linalg.inv(BGR_to_LMS_matrix) @ LMS_Deuteranopia_matrix @ BGR_to_LMS_matrix

Grayscale_matrix = np.array([
    [0.114,  0.587, 0.299],
    [0.114,  0.587, 0.299],
    [0.114,  0.587, 0.299],
])

class ColorFilter(enum.IntEnum):
    NONE = enum.auto()
    GRAYSCALE = enum.auto()
    SIMULATE_DEUTERANOPIA = enum.auto()

COLOR_FILTER_MATRIX = {
    ColorFilter.NONE: np.identity(3),
    ColorFilter.GRAYSCALE: Grayscale_matrix,
    ColorFilter.SIMULATE_DEUTERANOPIA: BGR_Deuteranopia_matrix,
}



def BGR_to_LMS(bgr):
    assert bgr.dtype == np.uint8
    lms = apply_color_matrix(bgr.astype(float) / 255, BGR_to_LMS_matrix)
    assert lms.dtype == float
    return lms


def LMS_to_BGR(lms):
    assert lms.dtype == np.float
    bgr = apply_color_matrix(lms, np.linalg.inv(BGR_to_LMS_matrix)) * 255
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    assert bgr.dtype == np.uint8
    return bgr


def apply_color_filter(bgr, color_filter: ColorFilter):
    assert bgr.dtype == np.uint8
    matrix = COLOR_FILTER_MATRIX[color_filter]
    res = cs.apply_color_matrix(bgr.astype(float), matrix)
    res = np.clip(res, 0, 255).astype(np.uint8)
    assert res.dtype == np.uint8
    return res

